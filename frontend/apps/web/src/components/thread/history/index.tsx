import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Skeleton } from "@/components/ui/skeleton";
import { useMediaQuery } from "@/hooks/useMediaQuery";
import { getBackendBaseUrl } from "@/lib/backend-base-url";
import { AppThread } from "@/lib/chat-types";
import { useThreads } from "@/providers/Thread";
import { PanelRightClose, PanelRightOpen, Trash2 } from "lucide-react";
import { useEffect } from "react";
import { parseAsBoolean, useQueryState } from "nuqs";
import { toast } from "sonner";

function ThreadList({
  threads,
  onThreadClick,
  onThreadDelete,
}: {
  threads: AppThread[];
  onThreadClick?: (threadId: string) => void;
  onThreadDelete?: (threadId: string) => void;
}) {
  const [threadId, setThreadId] = useQueryState("threadId");

  return (
    <div className="h-full flex flex-col w-full gap-2 items-start justify-start overflow-y-scroll [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-track]:bg-transparent">
      {threads.map((thread) => (
        <div key={thread.thread_id} className="w-full px-1">
          <div className="flex w-full items-start gap-2">
            <Button
              variant="ghost"
              className="text-left items-start justify-start font-normal w-[236px]"
              onClick={(e) => {
                e.preventDefault();
                onThreadClick?.(thread.thread_id);
                if (thread.thread_id === threadId) return;
                setThreadId(thread.thread_id);
              }}
            >
              <div className="flex w-full flex-col items-start">
                <p className="truncate text-ellipsis">{thread.title}</p>
                <span className="text-xs text-slate-500">
                  {new Date(thread.updated_at).toLocaleString()}
                </span>
              </div>
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="shrink-0 text-slate-500 hover:bg-rose-50 hover:text-rose-700"
              onClick={(e) => {
                e.preventDefault();
                e.stopPropagation();
                onThreadDelete?.(thread.thread_id);
              }}
              aria-label={`Delete thread ${thread.title}`}
              title="Delete thread"
            >
              <Trash2 className="size-4" />
            </Button>
          </div>
        </div>
      ))}
    </div>
  );
}

function ThreadHistoryLoading() {
  return (
    <div className="h-full flex flex-col w-full gap-2 items-start justify-start overflow-y-scroll [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-track]:bg-transparent">
      {Array.from({ length: 12 }).map((_, i) => (
        <Skeleton key={`skeleton-${i}`} className="w-[280px] h-10" />
      ))}
    </div>
  );
}

export default function ThreadHistory() {
  const backendBaseUrl = getBackendBaseUrl();
  const isLargeScreen = useMediaQuery("(min-width: 1024px)");
  const [chatHistoryOpen, setChatHistoryOpen] = useQueryState(
    "chatHistoryOpen",
    parseAsBoolean.withDefault(false),
  );
  const [, setThreadId] = useQueryState("threadId");

  const {
    deleteThread,
    getThreads,
    threads,
    setThreads,
    threadsLoading,
    setThreadsLoading,
  } = useThreads();

  useEffect(() => {
    let cancelled = false;
    setThreadsLoading(true);
    getThreads()
      .then((fetched) => {
        if (cancelled) return;
        setThreads(fetched);
      })
      .finally(() => {
        if (!cancelled) setThreadsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [getThreads, setThreads, setThreadsLoading]);

  const handleDeleteThread = async (threadId: string) => {
    try {
      const response = await fetch(`${backendBaseUrl}/api/v1/threads/${threadId}`, {
        method: "DELETE",
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
    } catch (error) {
      const detail =
        error instanceof Error ? error.message : "Unexpected thread delete error";
      toast.error("Backend thread delete failed", {
        description:
          "The local thread will still be removed from the UI history. " +
          `Backend detail: ${detail}`,
        richColors: true,
        closeButton: true,
      });
    } finally {
      deleteThread(threadId);
      setThreadId((current) => (current === threadId ? null : current));
    }
  };

  return (
    <>
      <div className="hidden lg:flex flex-col border-r-[1px] border-slate-300 items-start justify-start gap-6 h-screen w-[300px] shrink-0 shadow-inner-right">
        <div className="flex items-center justify-between w-full pt-1.5 px-4">
          <Button
            className="hover:bg-gray-100"
            variant="ghost"
            onClick={() => setChatHistoryOpen((p) => !p)}
          >
            {chatHistoryOpen ? (
              <PanelRightOpen className="size-5" />
            ) : (
              <PanelRightClose className="size-5" />
            )}
          </Button>
          <h1 className="text-xl font-semibold tracking-tight">
            Thread History
          </h1>
        </div>
        {threadsLoading ? (
          <ThreadHistoryLoading />
        ) : (
          <ThreadList threads={threads} onThreadDelete={handleDeleteThread} />
        )}
      </div>
      <div className="lg:hidden">
        <Sheet
          open={!!chatHistoryOpen && !isLargeScreen}
          onOpenChange={(open) => {
            if (isLargeScreen) return;
            setChatHistoryOpen(open);
          }}
        >
          <SheetContent side="left" className="lg:hidden flex">
            <SheetHeader>
              <SheetTitle>Thread History</SheetTitle>
            </SheetHeader>
            <ThreadList
              threads={threads}
              onThreadClick={() => setChatHistoryOpen((o) => !o)}
              onThreadDelete={handleDeleteThread}
            />
          </SheetContent>
        </Sheet>
      </div>
    </>
  );
}
