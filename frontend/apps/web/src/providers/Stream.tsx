import React, {
  createContext,
  useContext,
  ReactNode,
  useState,
  useEffect,
} from "react";
import { useStream } from "@langchain/langgraph-sdk/react";
import { type Message } from "@langchain/langgraph-sdk";
import { validate } from "uuid";
import {
  uiMessageReducer,
  type UIMessage,
  type RemoveUIMessage,
} from "@langchain/langgraph-sdk/react-ui";
import { useQueryState } from "nuqs";
import { getApiKey } from "@/lib/api-key";
import { useThreads } from "./Thread";
import { toast } from "sonner";
import { useSession } from "next-auth/react";
import { createClient } from "./client";

export type StateType = { messages: Message[]; ui?: UIMessage[] };

const useTypedStream = useStream<
  StateType,
  {
    UpdateType: {
      messages?: Message[] | Message | string;
      ui?: (UIMessage | RemoveUIMessage)[] | UIMessage | RemoveUIMessage;
    };
    CustomEventType: UIMessage | RemoveUIMessage;
  }
>;

type StreamContextType = ReturnType<typeof useTypedStream>;
const StreamContext = createContext<StreamContextType | undefined>(undefined);

async function sleep(ms = 4000) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function getContentPreview(content: unknown): string | null {
  if (typeof content === "string") {
    const trimmed = content.trim();
    return trimmed ? trimmed.slice(0, 80) : null;
  }

  if (Array.isArray(content)) {
    const textParts = content
      .filter(
        (part): part is { type: "text"; text: string } =>
          typeof part === "object" &&
          part !== null &&
          "type" in part &&
          "text" in part &&
          (part as any).type === "text" &&
          typeof (part as any).text === "string",
      )
      .map((part) => part.text.trim())
      .filter(Boolean);

    const joined = textParts.join(" ").trim();
    return joined ? joined.slice(0, 80) : null;
  }

  return null;
}

function getThreadSearchMetadata(
  assistantId: string,
): { graph_id: string } | { assistant_id: string } {
  if (validate(assistantId)) {
    return { assistant_id: assistantId };
  } else {
    return { graph_id: assistantId };
  }
}

async function setThreadTitleFromHistory(
  client: ReturnType<typeof createClient>,
  threadId: string,
  baseMetadata: Record<string, unknown>,
) {
  for (let attempt = 0; attempt < 5; attempt++) {
    try {
      const history = await client.threads.getHistory(threadId, { limit: 20 });
      const latestState = Array.isArray(history) ? history[0] : undefined;
      const messages = (latestState?.values as any)?.messages;
      if (!Array.isArray(messages) || messages.length === 0) {
        await sleep(700);
        continue;
      }

      const firstHuman = messages.find((m: any) => m?.type === "human") ?? messages[0];
      const title = getContentPreview(firstHuman?.content);
      if (!title) {
        await sleep(700);
        continue;
      }

      await client.threads.update(threadId, {
        metadata: {
          ...baseMetadata,
          title,
        },
      });
      return;
    } catch {
      await sleep(700);
    }
  }
}

async function checkGraphStatus(
  apiUrl: string,
  apiKey: string | null,
): Promise<boolean> {
  try {
    const res = await fetch(`${apiUrl}/info`, {
      ...(apiKey && {
        headers: {
          "X-Api-Key": apiKey,
        },
      }),
    });

    return res.ok;
  } catch (e) {
    console.error(e);
    return false;
  }
}

const StreamSession = ({
  children,
  apiKey,
  apiUrl,
  assistantId,
}: {
  children: ReactNode;
  apiKey: string | null;
  apiUrl: string;
  assistantId: string;
}) => {
  const { data: session } = useSession();
  const userId = session?.user?.id ?? "anonymous";
  const userEmail = (session?.user?.email ?? "").trim().toLowerCase();
  const stableUserKey = userEmail || userId;
  const [threadId, setThreadId] = useQueryState("threadId");
  const { getThreads, setThreads } = useThreads();
  const streamValue = useTypedStream({
    apiUrl,
    apiKey: apiKey ?? undefined,
    assistantId,
    threadId: threadId ?? null,
    onCustomEvent: (event, options) => {
      options.mutate((prev) => {
        const ui = uiMessageReducer(prev.ui ?? [], event);
        return { ...prev, ui };
      });
    },
    onThreadId: (id) => {
      setThreadId(id);

      // Persist thread ownership metadata so history can be filtered by authenticated user.
      const client = createClient(apiUrl, apiKey ?? undefined);
      const baseMetadata = {
        ...getThreadSearchMetadata(assistantId),
        user_id: userId,
        user_email: userEmail,
        user_key: stableUserKey,
      };

      client.threads
        .update(id, {
          metadata: baseMetadata,
        })
        .catch(console.error);

      // Add a human-readable title from the first human message.
      setThreadTitleFromHistory(client, id, baseMetadata)
        .catch(console.error);

      // Refetch threads list when thread ID changes.
      // Wait for some seconds before fetching so we're able to get the new thread that was created.
      sleep().then(() => getThreads().then(setThreads).catch(console.error));
    },
  });

  useEffect(() => {
    checkGraphStatus(apiUrl, apiKey).then((ok) => {
      if (!ok) {
        toast.error("Failed to connect to the conversation backend", {
          description: () => (
            <p>
              Please ensure the backend runtime is reachable at <code>{apiUrl}</code>{" "}
              and that your API key is correctly set if needed.
            </p>
          ),
          duration: 10000,
          richColors: true,
          closeButton: true,
        });
      }
    });
  }, [apiKey, apiUrl]);

  return (
    <StreamContext.Provider value={streamValue}>
      {children}
    </StreamContext.Provider>
  );
};

// Default values for the form
const DEFAULT_API_URL = "http://localhost:2024";
const DEFAULT_ASSISTANT_ID = "agent";

export const StreamProvider: React.FC<{ children: ReactNode }> = ({
  children,
}) => {
  const envApiUrl: string | undefined = process.env.NEXT_PUBLIC_API_URL;
  const envAssistantId: string | undefined =
    process.env.NEXT_PUBLIC_ASSISTANT_ID;
  const envApiKey: string | undefined =
    process.env.NEXT_PUBLIC_LANGSMITH_API_KEY;

  const [apiKey] = useState(() => {
    const storedKey = getApiKey();
    return storedKey || envApiKey || "";
  });
  const finalApiUrl = envApiUrl || DEFAULT_API_URL;
  const finalAssistantId = envAssistantId || DEFAULT_ASSISTANT_ID;

  return (
    <StreamSession
      apiKey={apiKey}
      apiUrl={finalApiUrl}
      assistantId={finalAssistantId}
    >
      {children}
    </StreamSession>
  );
};

// Create a custom hook to use the context
export const useStreamContext = (): StreamContextType => {
  const context = useContext(StreamContext);
  if (context === undefined) {
    throw new Error("useStreamContext must be used within a StreamProvider");
  }
  return context;
};

export default StreamContext;
