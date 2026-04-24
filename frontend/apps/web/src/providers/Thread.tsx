import { validate } from "uuid";
import { getApiKey } from "@/lib/api-key";
import { Thread } from "@langchain/langgraph-sdk";
import { useQueryState } from "nuqs";
import { useSession } from "next-auth/react";
import {
  createContext,
  useContext,
  ReactNode,
  useCallback,
  useEffect,
  useState,
  Dispatch,
  SetStateAction,
} from "react";
import { createClient } from "./client";

interface ThreadContextType {
  getThreads: () => Promise<Thread[]>;
  threads: Thread[];
  setThreads: Dispatch<SetStateAction<Thread[]>>;
  threadsLoading: boolean;
  setThreadsLoading: Dispatch<SetStateAction<boolean>>;
}

const ThreadContext = createContext<ThreadContextType | undefined>(undefined);

function getThreadSearchMetadata(
  assistantId: string,
): { graph_id: string } | { assistant_id: string } {
  if (validate(assistantId)) {
    return { assistant_id: assistantId };
  } else {
    return { graph_id: assistantId };
  }
}

function getUserAliases(stableUserKey: string): string[] {
  if (typeof window === "undefined" || !stableUserKey) return [];
  try {
    const raw = window.localStorage.getItem(`lg:user_aliases:${stableUserKey}`);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed.filter((x) => typeof x === "string") : [];
  } catch {
    return [];
  }
}

function saveUserAlias(stableUserKey: string, alias: string) {
  if (typeof window === "undefined" || !stableUserKey || !alias) return;
  const current = new Set(getUserAliases(stableUserKey));
  current.add(alias);
  window.localStorage.setItem(
    `lg:user_aliases:${stableUserKey}`,
    JSON.stringify(Array.from(current)),
  );
}

export function ThreadProvider({ children }: { children: ReactNode }) {
  const { data: session } = useSession();
  const envApiUrl: string | undefined = process.env.NEXT_PUBLIC_API_URL;
  const envAssistantId: string | undefined =
    process.env.NEXT_PUBLIC_ASSISTANT_ID;
  const [apiUrl] = useQueryState("apiUrl");
  const [assistantId] = useQueryState("assistantId");
  const finalApiUrl = apiUrl || envApiUrl;
  const finalAssistantId = assistantId || envAssistantId;
  const userId = session?.user?.id ?? "anonymous";
  const userEmail = (session?.user?.email ?? "").trim().toLowerCase();
  const stableUserKey = userEmail || userId;
  const [threads, setThreads] = useState<Thread[]>([]);
  const [threadsLoading, setThreadsLoading] = useState(false);

  useEffect(() => {
    saveUserAlias(stableUserKey, userId);
  }, [stableUserKey, userId]);

  const getThreads = useCallback(async (): Promise<Thread[]> => {
    if (!finalApiUrl || !finalAssistantId) return [];
    const client = createClient(finalApiUrl, getApiKey() ?? undefined);

    const allThreads = await client.threads.search({
      metadata: {
        ...getThreadSearchMetadata(finalAssistantId),
      },
      limit: 100,
    });

    const aliases = new Set(getUserAliases(stableUserKey));
    aliases.add(userId);

    const filtered = allThreads.filter((t: any) => {
      const ownerEmail =
        typeof t?.metadata?.user_email === "string"
          ? t.metadata.user_email.trim().toLowerCase()
          : "";
      const ownerKey =
        typeof t?.metadata?.user_key === "string" ? t.metadata.user_key : "";
      const ownerId =
        typeof t?.metadata?.user_id === "string" ? t.metadata.user_id : "";

      if (ownerEmail) return ownerEmail === userEmail;
      if (ownerKey) return ownerKey === stableUserKey;
      if (ownerId) return aliases.has(ownerId);
      return false;
    });

    return filtered;
  }, [finalApiUrl, finalAssistantId, stableUserKey, userEmail, userId]);

  const value = {
    getThreads,
    threads,
    setThreads,
    threadsLoading,
    setThreadsLoading,
  };

  return (
    <ThreadContext.Provider value={value}>{children}</ThreadContext.Provider>
  );
}

export function useThreads() {
  const context = useContext(ThreadContext);
  if (context === undefined) {
    throw new Error("useThreads must be used within a ThreadProvider");
  }
  return context;
}
