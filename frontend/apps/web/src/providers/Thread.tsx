import { AppThread } from "@/lib/chat-types";
import { useSession } from "next-auth/react";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type Dispatch,
  type ReactNode,
  type SetStateAction,
} from "react";

interface ThreadContextType {
  getThreads: () => Promise<AppThread[]>;
  getThreadById: (threadId: string | null) => AppThread | undefined;
  saveThread: (thread: AppThread) => void;
  deleteThread: (threadId: string) => void;
  threads: AppThread[];
  setThreads: Dispatch<SetStateAction<AppThread[]>>;
  threadsLoading: boolean;
  setThreadsLoading: Dispatch<SetStateAction<boolean>>;
}

const ThreadContext = createContext<ThreadContextType | undefined>(undefined);

function sortThreads(threads: AppThread[]): AppThread[] {
  return [...threads].sort((a, b) => {
    return (
      new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime()
    );
  });
}

function safeParseThreads(raw: string | null): AppThread[] {
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (thread): thread is AppThread =>
        typeof thread?.thread_id === "string" &&
        typeof thread?.title === "string" &&
        typeof thread?.updated_at === "string" &&
        Array.isArray(thread?.messages),
    );
  } catch {
    return [];
  }
}

export function ThreadProvider({ children }: { children: ReactNode }) {
  const { data: session } = useSession();
  const userId = session?.user?.id ?? "anonymous";
  const userEmail = (session?.user?.email ?? "").trim().toLowerCase();
  const stableUserKey = userEmail || userId;
  const storageKey = useMemo(
    () => `pfg:threads:${stableUserKey}`,
    [stableUserKey],
  );

  const [threads, setThreads] = useState<AppThread[]>([]);
  const [threadsLoading, setThreadsLoading] = useState(false);

  const getThreads = useCallback(async (): Promise<AppThread[]> => {
    if (typeof window === "undefined") return [];
    const parsed = safeParseThreads(window.localStorage.getItem(storageKey));
    return sortThreads(parsed);
  }, [storageKey]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    setThreadsLoading(true);
    getThreads()
      .then((loaded) => setThreads(loaded))
      .finally(() => setThreadsLoading(false));
  }, [getThreads]);

  const persistThreads = useCallback(
    (nextThreadsOrUpdater: AppThread[] | ((current: AppThread[]) => AppThread[])) => {
      const resolvedThreads =
        typeof nextThreadsOrUpdater === "function"
          ? nextThreadsOrUpdater(threads)
          : nextThreadsOrUpdater;
      const sortedThreads = sortThreads(resolvedThreads);
      setThreads(sortedThreads);
      if (typeof window !== "undefined") {
        window.localStorage.setItem(storageKey, JSON.stringify(sortedThreads));
      }
    },
    [storageKey, threads],
  );

  const getThreadById = useCallback(
    (threadId: string | null) =>
      threads.find((thread) => thread.thread_id === threadId),
    [threads],
  );

  const saveThread = useCallback(
    (thread: AppThread) => {
      persistThreads((currentThreads) => {
        const nextThreads = currentThreads.filter(
          (existingThread) => existingThread.thread_id !== thread.thread_id,
        );
        nextThreads.push(thread);
        return nextThreads;
      });
    },
    [persistThreads],
  );

  const deleteThread = useCallback(
    (threadId: string) => {
      persistThreads((currentThreads) =>
        currentThreads.filter((thread) => thread.thread_id !== threadId),
      );
    },
    [persistThreads],
  );

  const value = {
    getThreads,
    getThreadById,
    saveThread,
    deleteThread,
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
  if (!context) {
    throw new Error("useThreads must be used within a ThreadProvider");
  }
  return context;
}
