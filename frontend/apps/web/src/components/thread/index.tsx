import { v4 as uuidv4 } from "uuid";
import { ReactNode, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { useStreamContext } from "@/providers/Stream";
import { useState, FormEvent } from "react";
import { Button } from "../ui/button";
import { Checkpoint, Message } from "@langchain/langgraph-sdk";
import { AssistantMessage, AssistantMessageLoading } from "./messages/ai";
import { HumanMessage } from "./messages/human";
import {
  DO_NOT_RENDER_ID_PREFIX,
  ensureToolCallsHaveResponses,
} from "@/lib/ensure-tool-responses";
import { LangGraphLogoSVG } from "../icons/langgraph";
import { TooltipIconButton } from "./tooltip-icon-button";
import {
  ArrowDown,
  LoaderCircle,
  PanelRightOpen,
  PanelRightClose,
  SquarePen,
  LogOut,
} from "lucide-react";
import { useQueryState, parseAsBoolean } from "nuqs";
import { StickToBottom, useStickToBottomContext } from "use-stick-to-bottom";
import ThreadHistory from "./history";
import { toast } from "sonner";
import { useMediaQuery } from "@/hooks/useMediaQuery";
import { Label } from "../ui/label";
import { Switch } from "../ui/switch";
import { GitHubSVG } from "../icons/github";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "../ui/tooltip";
import { useSession, signOut } from "next-auth/react";

function StickyToBottomContent(props: {
  content: ReactNode;
  footer?: ReactNode;
  className?: string;
  contentClassName?: string;
}) {
  const context = useStickToBottomContext();
  return (
    <div
      ref={context.scrollRef}
      style={{ width: "100%", height: "100%" }}
      className={props.className}
    >
      <div ref={context.contentRef} className={props.contentClassName}>
        {props.content}
      </div>

      {props.footer}
    </div>
  );
}

function ScrollToBottom(props: { className?: string }) {
  const { isAtBottom, scrollToBottom } = useStickToBottomContext();

  if (isAtBottom) return null;
  return (
    <Button
      variant="outline"
      className={props.className}
      onClick={() => scrollToBottom()}
    >
      <ArrowDown className="w-4 h-4" />
      <span>Scroll to bottom</span>
    </Button>
  );
}

function OpenGitHubRepo() {
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <a
            href="https://github.com/langchain-ai/agent-chat-ui"
            target="_blank"
            className="flex items-center justify-center"
          >
            <GitHubSVG width="24" height="24" />
          </a>
        </TooltipTrigger>
        <TooltipContent side="left">
          <p>Open GitHub repo</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

export function Thread() {
  const [threadId, setThreadId] = useQueryState("threadId");
  const [chatHistoryOpen, setChatHistoryOpen] = useQueryState(
    "chatHistoryOpen",
    parseAsBoolean.withDefault(false),
  );
  const [hideToolCalls, setHideToolCalls] = useQueryState(
    "hideToolCalls",
    parseAsBoolean.withDefault(false),
  );
  const [dataset, setDataset] = useQueryState("dataset", { defaultValue: "movielens" });
  const [recModel, setRecModel] = useQueryState("rec_model", { defaultValue: "mf" });

  const { data: session } = useSession();

  const [input, setInput] = useState("");
  const [firstTokenReceived, setFirstTokenReceived] = useState(false);
  const isLargeScreen = useMediaQuery("(min-width: 1024px)");

  const stream = useStreamContext();
  const messages = stream.messages;
  const isLoading = stream.isLoading;
  const inFlightTrace = useRef<{ traceId: string; startedAtMs: number } | null>(
    null,
  );

  const lastError = useRef<string | undefined>(undefined);

  useEffect(() => {
    if (!stream.error) {
      lastError.current = undefined;
      return;
    }
    try {
      const message = (stream.error as any).message;
      if (!message || lastError.current === message) {
        // Message has already been logged. do not modify ref, return early.
        return;
      }

      // Message is defined, and it has not been logged yet. Save it, and send the error
      lastError.current = message;
      toast.error("An error occurred. Please try again.", {
        description: (
          <p>
            <strong>Error:</strong> <code>{message}</code>
          </p>
        ),
        richColors: true,
        closeButton: true,
      });
    } catch {
      // no-op
    }
  }, [stream.error]);

  // TODO: this should be part of the useStream hook
  const prevMessageLength = useRef(0);
  useEffect(() => {
    if (
      messages.length !== prevMessageLength.current &&
      messages?.length &&
      messages[messages.length - 1].type === "ai"
    ) {
      setFirstTokenReceived(true);
    }

    prevMessageLength.current = messages.length;
  }, [messages]);

  useEffect(() => {
    if (!isLoading && inFlightTrace.current) {
      const elapsedMs = performance.now() - inFlightTrace.current.startedAtMs;
      console.info("[frontend][response_complete]", {
        trace_id: inFlightTrace.current.traceId,
        elapsed_ms: Number(elapsedMs.toFixed(2)),
        total_messages: messages.length,
      });
      inFlightTrace.current = null;
    }
  }, [isLoading, messages.length]);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;
    setFirstTokenReceived(false);

    const newHumanMessage: Message = {
      id: uuidv4(),
      type: "human",
      content: input,
    };

    const toolMessages = ensureToolCallsHaveResponses(stream.messages);
    const stableUserId =
      session?.user?.email?.trim().toLowerCase() ??
      session?.user?.id ??
      "anonymous";
    const traceId = uuidv4();
    inFlightTrace.current = {
      traceId,
      startedAtMs: performance.now(),
    };
    console.info("[frontend][submit]", {
      trace_id: traceId,
      dataset,
      rec_model: recModel,
      user_id: stableUserId,
      thread_id: threadId ?? null,
      input_preview: input.slice(0, 120),
    });
    stream.submit(
      { messages: [...toolMessages, newHumanMessage] },
      {
        streamMode: ["values"],
        config: {
          configurable: {
            dataset,
            rec_model: recModel,
            user_id: stableUserId,
            trace_id: traceId,
          }
        },
        optimisticValues: (prev) => ({
          ...prev,
          messages: [
            ...(prev.messages ?? []),
            ...toolMessages,
            newHumanMessage,
          ],
        }),
      },
    );

    setInput("");
  };

  const handleRegenerate = (
    parentCheckpoint: Checkpoint | null | undefined,
  ) => {
    // Do this so the loading state is correct
    prevMessageLength.current = prevMessageLength.current - 1;
    setFirstTokenReceived(false);
    stream.submit(undefined, {
      checkpoint: parentCheckpoint,
      streamMode: ["values"],
    });
  };

  const chatStarted = !!threadId || !!messages.length;
  const hasNoAIOrToolMessages = !messages.find(
    (m) => m.type === "ai" || m.type === "tool",
  );

  return (
    <div className="flex w-full h-screen overflow-hidden">
      <div className="relative lg:flex hidden">
        <motion.div
          className="absolute h-full border-r bg-white overflow-hidden z-20"
          style={{ width: 300 }}
          animate={
            isLargeScreen
              ? { x: chatHistoryOpen ? 0 : -300 }
              : { x: chatHistoryOpen ? 0 : -300 }
          }
          initial={{ x: -300 }}
          transition={
            isLargeScreen
              ? { type: "spring", stiffness: 300, damping: 30 }
              : { duration: 0 }
          }
        >
          <div className="relative h-full" style={{ width: 300 }}>
            <ThreadHistory />
          </div>
        </motion.div>
      </div>
      <motion.div
        className={cn(
          "flex-1 flex flex-col min-w-0 overflow-hidden relative",
          !chatStarted && "grid-rows-[1fr]",
        )}
        layout={isLargeScreen}
        animate={{
          marginLeft: chatHistoryOpen ? (isLargeScreen ? 300 : 0) : 0,
          width: chatHistoryOpen
            ? isLargeScreen
              ? "calc(100% - 300px)"
              : "100%"
            : "100%",
        }}
        transition={
          isLargeScreen
            ? { type: "spring", stiffness: 300, damping: 30 }
            : { duration: 0 }
        }
      >
        {!chatStarted && (
          <div className="absolute top-0 left-0 w-full flex items-center justify-between gap-3 p-2 pl-4 z-10">
            <div>
              {(!chatHistoryOpen || !isLargeScreen) && (
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
              )}
            </div>
            <div className="absolute top-2 right-4 flex items-center gap-3">
              <OpenGitHubRepo />
              {session?.user && (
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <button
                        id="logout-button"
                        onClick={() => signOut({ callbackUrl: "/login" })}
                        className="flex items-center gap-2 rounded-full border border-gray-200 bg-white px-3 py-1.5 text-sm text-gray-700 shadow-sm transition hover:bg-gray-50 hover:shadow-md"
                      >
                        {session.user.image ? (
                          <img
                            src={session.user.image}
                            alt={session.user.name ?? "User"}
                            className="h-6 w-6 rounded-full"
                          />
                        ) : (
                          <span className="flex h-6 w-6 items-center justify-center rounded-full bg-gradient-to-br from-purple-500 to-blue-500 text-xs font-bold text-white">
                            {session.user.name?.[0]?.toUpperCase() ?? "U"}
                          </span>
                        )}
                        <LogOut className="h-3.5 w-3.5 text-gray-400" />
                      </button>
                    </TooltipTrigger>
                    <TooltipContent side="bottom">
                      <p>Cerrar sesión ({session.user.name})</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              )}
            </div>
          </div>
        )}
        {chatStarted && (
          <div className="flex items-center justify-between gap-3 p-2 z-10 relative">
            <div className="flex items-center justify-start gap-2 relative">
              <div className="absolute left-0 z-10">
                {(!chatHistoryOpen || !isLargeScreen) && (
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
                )}
              </div>
              <motion.button
                className="flex gap-2 items-center cursor-pointer"
                onClick={() => setThreadId(null)}
                animate={{
                  marginLeft: !chatHistoryOpen ? 48 : 0,
                }}
                transition={{
                  type: "spring",
                  stiffness: 300,
                  damping: 30,
                }}
              >
                <LangGraphLogoSVG width={32} height={32} />
                <span className="text-xl font-semibold tracking-tight">
                  Recommender Chatbot
                </span>
              </motion.button>

              {/* Desktop Model/Dataset Selectors */}
              <div className="hidden lg:flex ml-4 gap-2">
                <select
                  className="px-3 py-1 text-sm border bg-white rounded-md shadow-sm outline-none cursor-pointer text-gray-700"
                  value={dataset || "movielens"}
                  onChange={(e) => setDataset(e.target.value)}
                >
                  <option value="movielens">🎥 MovieLens</option>
                  <option value="amazon_electronics">🛒 Amazon</option>
                  <option value="yelp">🍔 Yelp</option>
                  <option value="lastfm">🎵 LastFM</option>
                  <option value="foursquare">📍 Foursquare</option>
                </select>
                <select
                  className="px-3 py-1 text-sm border bg-white rounded-md shadow-sm outline-none cursor-pointer text-gray-700"
                  value={recModel || "mf"}
                  onChange={(e) => setRecModel(e.target.value)}
                >
                  <option value="mf">Matrix Factorization</option>
                  <option value="two_tower">Two-Tower</option>
                  <option value="sasrec">SASRec</option>
                  <option value="llm_rag">LLM + RAG</option>
                </select>
              </div>

            </div>

            <div className="flex items-center gap-4">
              <div className="flex items-center">
                <OpenGitHubRepo />
              </div>
              <TooltipIconButton
                size="lg"
                className="p-4"
                tooltip="New thread"
                variant="ghost"
                onClick={() => setThreadId(null)}
              >
                <SquarePen className="size-5" />
              </TooltipIconButton>

              {/* User avatar + logout */}
              {session?.user && (
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <button
                        id="logout-button"
                        onClick={() => signOut({ callbackUrl: "/login" })}
                        className="flex items-center gap-2 rounded-full border border-gray-200 bg-white px-3 py-1.5 text-sm text-gray-700 shadow-sm transition hover:bg-gray-50 hover:shadow-md"
                      >
                        {session.user.image ? (
                          <img
                            src={session.user.image}
                            alt={session.user.name ?? "User"}
                            className="h-6 w-6 rounded-full"
                          />
                        ) : (
                          <span className="flex h-6 w-6 items-center justify-center rounded-full bg-gradient-to-br from-purple-500 to-blue-500 text-xs font-bold text-white">
                            {session.user.name?.[0]?.toUpperCase() ?? "U"}
                          </span>
                        )}
                        <LogOut className="h-3.5 w-3.5 text-gray-400" />
                      </button>
                    </TooltipTrigger>
                    <TooltipContent side="bottom">
                      <p>Cerrar sesión ({session.user.name})</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              )}
            </div>

            <div className="absolute inset-x-0 top-full h-5 bg-gradient-to-b from-background to-background/0" />
          </div>
        )}

        <StickToBottom className="relative flex-1 overflow-hidden">
          <StickyToBottomContent
            className={cn(
              "absolute px-4 inset-0 overflow-y-scroll [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-track]:bg-transparent",
              !chatStarted && "flex flex-col items-stretch mt-[25vh]",
              chatStarted && "grid grid-rows-[1fr_auto]",
            )}
            contentClassName="pt-8 pb-16  max-w-3xl mx-auto flex flex-col gap-4 w-full"
            content={
              <>
                {messages
                  .filter((m) => !m.id?.startsWith(DO_NOT_RENDER_ID_PREFIX))
                  .map((message, index) =>
                    message.type === "human" ? (
                      <HumanMessage
                        key={message.id || `${message.type}-${index}`}
                        message={message}
                        isLoading={isLoading}
                      />
                    ) : (
                      <AssistantMessage
                        key={message.id || `${message.type}-${index}`}
                        message={message}
                        isLoading={isLoading}
                        handleRegenerate={handleRegenerate}
                      />
                    ),
                  )}
                {/* Special rendering case where there are no AI/tool messages, but there is an interrupt.
                    We need to render it outside of the messages list, since there are no messages to render */}
                {hasNoAIOrToolMessages && !!stream.interrupt && (
                  <AssistantMessage
                    key="interrupt-msg"
                    message={undefined}
                    isLoading={isLoading}
                    handleRegenerate={handleRegenerate}
                  />
                )}
                {isLoading && !firstTokenReceived && (
                  <AssistantMessageLoading />
                )}
              </>
            }
            footer={
              <div className="sticky flex flex-col items-center gap-8 bottom-0 bg-white">
                {!chatStarted && (
                  <div className="flex flex-col items-center gap-4">
                    <div className="flex gap-3 items-center">
                      <LangGraphLogoSVG className="flex-shrink-0 h-8" />
                      <h1 className="text-2xl font-semibold tracking-tight">
                        PFG Recommender System
                      </h1>
                    </div>

                    {/* Selectors for Home screen */}
                    <div className="flex flex-wrap items-center justify-center gap-3">
                      <select
                        className="px-4 py-2 text-sm border bg-white rounded-md shadow-sm outline-none cursor-pointer text-gray-700"
                        value={dataset || "movielens"}
                        onChange={(e) => setDataset(e.target.value)}
                      >
                        <option value="movielens">🎥 MovieLens</option>
                        <option value="amazon_electronics">🛒 Amazon</option>
                        <option value="yelp">🍔 Yelp</option>
                        <option value="lastfm">🎵 LastFM</option>
                        <option value="foursquare">📍 Foursquare</option>
                      </select>
                      <select
                        className="px-4 py-2 text-sm border bg-white rounded-md shadow-sm outline-none cursor-pointer text-gray-700"
                        value={recModel || "mf"}
                        onChange={(e) => setRecModel(e.target.value)}
                      >
                        <option value="mf">Matrix Factorization</option>
                        <option value="two_tower">Two-Tower</option>
                        <option value="sasrec">SASRec</option>
                        <option value="llm_rag">LLM + RAG</option>
                      </select>
                    </div>
                  </div>
                )}

                <ScrollToBottom className="absolute bottom-full left-1/2 -translate-x-1/2 mb-4 animate-in fade-in-0 zoom-in-95" />

                <div className="bg-muted rounded-2xl border shadow-xs mx-auto mb-8 w-full max-w-3xl relative z-10">
                  <form
                    onSubmit={handleSubmit}
                    className="grid grid-rows-[1fr_auto] gap-2 max-w-3xl mx-auto"
                  >
                    <textarea
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      onKeyDown={(e) => {
                        if (
                          e.key === "Enter" &&
                          !e.shiftKey &&
                          !e.metaKey &&
                          !e.nativeEvent.isComposing
                        ) {
                          e.preventDefault();
                          const el = e.target as HTMLElement | undefined;
                          const form = el?.closest("form");
                          form?.requestSubmit();
                        }
                      }}
                      placeholder="Type your message..."
                      className="p-3.5 pb-0 border-none bg-transparent field-sizing-content shadow-none ring-0 outline-none focus:outline-none focus:ring-0 resize-none"
                    />

                    <div className="flex items-center justify-between p-2 pt-4">
                      <div>
                        <div className="flex items-center space-x-2">
                          <Switch
                            id="render-tool-calls"
                            checked={hideToolCalls ?? false}
                            onCheckedChange={setHideToolCalls}
                          />
                          <Label
                            htmlFor="render-tool-calls"
                            className="text-sm text-gray-600"
                          >
                            Hide Tool Calls
                          </Label>
                        </div>
                      </div>
                      {stream.isLoading ? (
                        <Button key="stop" onClick={() => stream.stop()}>
                          <LoaderCircle className="w-4 h-4 animate-spin" />
                          Cancel
                        </Button>
                      ) : (
                        <Button
                          type="submit"
                          className="transition-all shadow-md"
                          disabled={isLoading || !input.trim()}
                        >
                          Send
                        </Button>
                      )}
                    </div>
                  </form>
                </div>
              </div>
            }
          />
        </StickToBottom>
      </motion.div>
    </div>
  );
}
