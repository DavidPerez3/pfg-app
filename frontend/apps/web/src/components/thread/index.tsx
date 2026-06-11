"use client";

import {
  DatasetUserOption,
  AppMessage,
  AppRecModel,
  AppThread,
} from "@/lib/chat-types";
import { cn } from "@/lib/utils";
import {
  attachResultToLatestAssistant,
  makeThreadTitle,
  mergeBackendMessages,
} from "@/lib/thread-helpers";
import { getBackendBaseUrl } from "@/lib/backend-base-url";
import { useThreads } from "@/providers/Thread";
import { useSession, signOut } from "next-auth/react";
import { parseAsBoolean, useQueryState } from "nuqs";
import React, { FormEvent, ReactNode, useMemo, useRef, useState } from "react";
import { v4 as uuidv4 } from "uuid";
import { motion } from "framer-motion";
import { StickToBottom, useStickToBottomContext } from "use-stick-to-bottom";
import {
  ArrowDown,
  LoaderCircle,
  LogOut,
  PanelRightClose,
  PanelRightOpen,
  SquarePen,
} from "lucide-react";

import { AssistantMessage, AssistantMessageLoading } from "./messages/ai";
import { HumanMessage } from "./messages/human";
import ThreadHistory from "./history";
import { LangGraphLogoSVG } from "../icons/langgraph";
import { TooltipIconButton } from "./tooltip-icon-button";
import { Button } from "../ui/button";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "../ui/tooltip";
import { useMediaQuery } from "@/hooks/useMediaQuery";
import { toast } from "sonner";

type FeedbackState = "idle" | "submitting" | "submitted";
const REC_MODEL_STORAGE_KEY = "pfg:selected-rec-model";
const DATASET_STORAGE_KEY = "pfg:selected-dataset";
const DATASET_USERS_STORAGE_KEY = "pfg:selected-dataset-users";
const REC_MODEL_OPTIONS: Array<{ value: AppRecModel; label: string }> = [
  { value: "mf", label: "Matrix Factorization" },
  { value: "two_tower", label: "Two-Tower" },
  { value: "two_tower_wide_deep", label: "Two-Tower + Wide&Deep" },
  { value: "sasrec", label: "SASRec" },
  { value: "llm_rag", label: "LLM + RAG" },
];
const DATASET_OPTIONS = [
  { value: "movielens", label: "MovieLens" },
  { value: "lastfm", label: "LastFM" },
  { value: "yelp", label: "Yelp" },
  { value: "amazon_electronics", label: "Amazon Electronics" },
  { value: "foursquare", label: "Foursquare" },
] as const;

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

export function Thread() {
  const backendBaseUrl = getBackendBaseUrl();
  const [threadId, setThreadId] = useQueryState("threadId");
  const [chatHistoryOpen, setChatHistoryOpen] = useQueryState(
    "chatHistoryOpen",
    parseAsBoolean.withDefault(false),
  );

  const { data: session } = useSession();
  const { threads, saveThread } = useThreads();
  const isLargeScreen = useMediaQuery("(min-width: 1024px)");

  const currentThread = useMemo(
    () => threads.find((thread) => thread.thread_id === threadId),
    [threadId, threads],
  );
  const messages = currentThread?.messages ?? [];
  const chatStarted = messages.length > 0;

  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [selectedRecModel, setSelectedRecModel] = useState<AppRecModel>("mf");
  const [selectedDataset, setSelectedDataset] = useState("movielens");
  const [datasetUsers, setDatasetUsers] = useState<DatasetUserOption[]>([]);
  const [datasetUsersLoading, setDatasetUsersLoading] = useState(false);
  const [selectedDatasetUsersByDataset, setSelectedDatasetUsersByDataset] =
    useState<Record<string, string>>({});
  const [feedbackByMessage, setFeedbackByMessage] = useState<
    Record<string, FeedbackState>
  >({});
  const abortRef = useRef<AbortController | null>(null);
  const hasStoredDatasetPreference = useRef(false);
  const hasManualDatasetSelection = useRef(false);

  const selectedDatasetUserId =
    selectedDatasetUsersByDataset[selectedDataset] ?? "";
  const selectedDatasetLabel =
    DATASET_OPTIONS.find((option) => option.value === selectedDataset)?.label ??
    selectedDataset;
  const selectedDatasetUserSummary = useMemo(
    () =>
      datasetUsers.find(
        (user) => String(user.user_id) === String(selectedDatasetUserId),
      ) ?? null,
    [datasetUsers, selectedDatasetUserId],
  );

  const stableUserId =
    session?.user?.email?.trim().toLowerCase() ??
    session?.user?.id ??
    "anonymous";

  React.useEffect(() => {
    if (typeof window === "undefined") return;
    const savedModel = window.localStorage.getItem(REC_MODEL_STORAGE_KEY);
    const isKnownModel = REC_MODEL_OPTIONS.some(
      (option) => option.value === savedModel,
    );
    if (isKnownModel) {
      setSelectedRecModel(savedModel as AppRecModel);
    }
  }, [backendBaseUrl]);

  React.useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(REC_MODEL_STORAGE_KEY, selectedRecModel);
  }, [selectedRecModel]);

  React.useEffect(() => {
    if (typeof window === "undefined") return;
    const savedDataset = window.localStorage.getItem(DATASET_STORAGE_KEY);
    const isKnownDataset = DATASET_OPTIONS.some(
      (option) => option.value === savedDataset,
    );
    if (savedDataset && isKnownDataset) {
      setSelectedDataset(savedDataset);
      hasStoredDatasetPreference.current = true;
    }

    const savedDatasetUsers = window.localStorage.getItem(
      DATASET_USERS_STORAGE_KEY,
    );
    if (!savedDatasetUsers) return;

    try {
      const parsed = JSON.parse(savedDatasetUsers) as Record<string, string>;
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        setSelectedDatasetUsersByDataset(parsed);
      }
    } catch {
      // Ignore malformed local storage values and keep defaults.
    }
  }, []);

  React.useEffect(() => {
    let cancelled = false;

    const loadBackendDefaults = async () => {
      try {
        const response = await fetch(`${backendBaseUrl}/`);
        if (!response.ok) return;
        const payload = await response.json();
        const nextDataset = payload?.defaults?.dataset;
        if (!cancelled && typeof nextDataset === "string" && nextDataset.trim()) {
          if (
            !hasStoredDatasetPreference.current &&
            !hasManualDatasetSelection.current
          ) {
            setSelectedDataset(nextDataset);
          }
        }
      } catch {
        // Keep the local default when backend metadata is unavailable.
      }
    };

    void loadBackendDefaults();
    return () => {
      cancelled = true;
    };
  }, []);

  React.useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(DATASET_STORAGE_KEY, selectedDataset);
  }, [backendBaseUrl, selectedDataset]);

  React.useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(
      DATASET_USERS_STORAGE_KEY,
      JSON.stringify(selectedDatasetUsersByDataset),
    );
  }, [selectedDatasetUsersByDataset]);

  React.useEffect(() => {
    let cancelled = false;

    const loadDatasetUsers = async () => {
      setDatasetUsersLoading(true);
      try {
        const requestStartedAt = performance.now();
        const params = new URLSearchParams({
          limit: "25",
          rec_model: selectedRecModel,
        });
        const response = await fetch(
          `${backendBaseUrl}/api/v1/datasets/${selectedDataset}/users?${params.toString()}`,
        );
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const payload = await response.json();
        if (cancelled) return;
        const users: DatasetUserOption[] = Array.isArray(payload?.users)
          ? payload.users
          : [];
        setDatasetUsers(users);
        console.info("[LATENCY][dataset-users]", {
          dataset: selectedDataset,
          frontend_roundtrip_ms: Number((performance.now() - requestStartedAt).toFixed(2)),
          backend_total_ms: payload?.latency?.backend_total_ms ?? null,
          backend_to_recommender_http_ms:
            payload?.latency?.backend_to_recommender_http_ms ?? null,
          recommender_total_ms: payload?.latency?.recommender_total_ms ?? null,
        });
      } catch {
        if (!cancelled) {
          setDatasetUsers([]);
          toast.error("Dataset users could not be loaded", {
            description:
              "The selector fell back to cold start because the frontend could not fetch the dataset-user list.",
            richColors: true,
            closeButton: true,
          });
        }
      } finally {
        if (!cancelled) {
          setDatasetUsersLoading(false);
        }
      }
    };

    void loadDatasetUsers();
    return () => {
      cancelled = true;
    };
  }, [backendBaseUrl, selectedDataset, selectedRecModel]);

  React.useEffect(() => {
    if (!selectedDatasetUserId) return;
    const exists = datasetUsers.some(
      (user) => String(user.user_id) === String(selectedDatasetUserId),
    );
    if (!exists) {
      setSelectedDatasetUsersByDataset((prev) => {
        const next = { ...prev };
        delete next[selectedDataset];
        return next;
      });
    }
  }, [datasetUsers, selectedDataset, selectedDatasetUserId]);

  const sendPrompt = async (rawPrompt: string) => {
    if (!rawPrompt.trim() || isLoading) return;

    const resolvedThreadId = currentThread?.thread_id ?? uuidv4();
    const now = new Date().toISOString();
    const traceId = uuidv4();
    const userMessage: AppMessage = {
      id: uuidv4(),
      role: "user",
      content: rawPrompt.trim(),
      createdAt: now,
      traceId,
    };

    const optimisticMessages = [...messages, userMessage];
    const optimisticThread: AppThread = {
      thread_id: resolvedThreadId,
      title: currentThread?.title || makeThreadTitle(userMessage.content),
      updated_at: now,
      messages: optimisticMessages,
    };

    if (!threadId) {
      setThreadId(resolvedThreadId);
    }
    saveThread(optimisticThread);
    setInput("");
    setIsLoading(true);

    abortRef.current = new AbortController();

    try {
      const clientSentAtMs = Date.now();
      const requestStartedAt = performance.now();
      const response = await fetch(
        `${backendBaseUrl}/api/v1/threads/${resolvedThreadId}/messages`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-Trace-Id": traceId,
          },
          signal: abortRef.current.signal,
          body: JSON.stringify({
            messages: optimisticMessages.map((message) => ({
              role: message.role,
              content: message.content,
            })),
            thread_id: resolvedThreadId,
            user_id: stableUserId,
            trace_id: traceId,
            client_sent_at_ms: clientSentAtMs,
            dataset: selectedDataset,
            rec_model: selectedRecModel,
            dataset_user_id: selectedDatasetUserId || null,
          }),
        },
      );

      if (!response.ok) {
        const detail = await response.text();
        throw new Error(detail || `HTTP ${response.status}`);
      }

      const payload = await response.json();
      console.info("[LATENCY][chat]", {
        trace_id: traceId,
        dataset: selectedDataset,
        rec_model: selectedRecModel,
        frontend_roundtrip_ms: Number((performance.now() - requestStartedAt).toFixed(2)),
        client_to_backend_ms: payload?.latency?.client_to_backend_ms ?? null,
        backend_total_ms: payload?.latency?.backend_total_ms ?? null,
        backend_to_recommender_http_ms:
          payload?.latency?.backend_to_recommender_http_ms ?? null,
        recommender_total_ms: payload?.latency?.recommender_total_ms ?? null,
      });
      const nextMessages = attachResultToLatestAssistant(
        mergeBackendMessages(
          optimisticMessages,
          payload.messages ?? [],
          payload.trace_id,
        ),
        payload.result,
      );

      saveThread({
        thread_id: resolvedThreadId,
        title: optimisticThread.title,
        updated_at: new Date().toISOString(),
        messages: nextMessages.length > 0 ? nextMessages : optimisticMessages,
      });
    } catch (error) {
      if (error instanceof Error && error.name === "AbortError") {
        return;
      }

      const detail =
        error instanceof Error ? error.message : "Unexpected chat error";
      const errorMessage: AppMessage = {
        id: uuidv4(),
        role: "assistant",
        content:
          "The backend request failed. Please check that the public FastAPI backend is running on `http://localhost:8000`.\n\n" +
          `Error: ${detail}`,
        createdAt: new Date().toISOString(),
        traceId,
      };

      saveThread({
        thread_id: resolvedThreadId,
        title: optimisticThread.title,
        updated_at: new Date().toISOString(),
        messages: [...optimisticMessages, errorMessage],
      });
      toast.error("Chat request failed", {
        description: detail,
        richColors: true,
        closeButton: true,
      });
    } finally {
      setIsLoading(false);
      abortRef.current = null;
    }
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    await sendPrompt(input);
  };

  const handleFollowUpPrompt = async (prompt: string) => {
    await sendPrompt(prompt);
  };

  const handleCancel = () => {
    abortRef.current?.abort();
    abortRef.current = null;
    setIsLoading(false);
  };

  const handleFeedback = async (
    message: AppMessage,
    rating: number,
    messageIndex: number,
  ) => {
    if (!currentThread?.thread_id) return;

    setFeedbackByMessage((prev) => ({
      ...prev,
      [message.id]: "submitting",
    }));

    try {
      const response = await fetch(
        `${backendBaseUrl}/api/v1/threads/${currentThread.thread_id}/feedback`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            user_id: stableUserId,
            thread_id: currentThread.thread_id,
            rating,
            comment:
              rating >= 4
                ? "Marked as useful from the chat UI."
                : "Marked as needing work from the chat UI.",
            message_index: messageIndex,
            trace_id: message.traceId,
          }),
        },
      );

      if (!response.ok) {
        const detail = await response.text();
        throw new Error(detail || `HTTP ${response.status}`);
      }

      setFeedbackByMessage((prev) => ({
        ...prev,
        [message.id]: "submitted",
      }));
      toast.success("Feedback stored", {
        description: "The backend recorded your rating for this answer.",
        richColors: true,
        closeButton: true,
      });
    } catch (error) {
      const detail =
        error instanceof Error ? error.message : "Unexpected feedback error";
      setFeedbackByMessage((prev) => ({
        ...prev,
        [message.id]: "idle",
      }));
      toast.error("Feedback request failed", {
        description: detail,
        richColors: true,
        closeButton: true,
      });
    }
  };

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      <div
        className={cn(
          "relative transition-all",
          chatHistoryOpen ? "w-[300px]" : "w-0",
        )}
      >
        <motion.div
          className="h-full"
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
        className="flex-1 flex flex-col min-w-0 overflow-hidden relative"
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
                      <p>Sign out ({session.user.name})</p>
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
                  PFG Recommender System
                </span>
              </motion.button>
            </div>

            <div className="flex items-center gap-4">
              <a
                href="/arch"
                className="hidden rounded-full border border-gray-200 bg-white px-3 py-1.5 text-sm font-medium text-gray-700 shadow-sm transition hover:bg-gray-50 lg:inline-flex"
              >
                Architecture
              </a>
              <a
                href="/api-reference"
                className="hidden rounded-full border border-gray-200 bg-white px-3 py-1.5 text-sm font-medium text-gray-700 shadow-sm transition hover:bg-gray-50 lg:inline-flex"
              >
                API
              </a>
              <TooltipIconButton
                size="lg"
                className="p-4"
                tooltip="New thread"
                variant="ghost"
                onClick={() => setThreadId(null)}
              >
                <SquarePen className="size-5" />
              </TooltipIconButton>
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
                      <p>Sign out ({session.user.name})</p>
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
            contentClassName="pt-8 pb-16 max-w-3xl mx-auto flex flex-col gap-4 w-full"
            content={
              <>
                {messages.map((message) =>
                  message.role === "user" ? (
                    <HumanMessage
                      key={message.id}
                      message={message}
                      isLoading={isLoading}
                    />
                  ) : (
                    <AssistantMessage
                      key={message.id}
                      message={message}
                      isLoading={isLoading}
                      onFollowUpPrompt={handleFollowUpPrompt}
                      feedbackState={feedbackByMessage[message.id] ?? "idle"}
                      onFeedback={(rating) =>
                        handleFeedback(
                          message,
                          rating,
                          messages.findIndex(
                            (currentMessage) =>
                              currentMessage.id === message.id,
                          ),
                        )
                      }
                    />
                  ),
                )}
                {isLoading && <AssistantMessageLoading />}
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
                    <p className="max-w-xl text-center text-sm text-gray-600">
                      Ask naturally for recommendations, similar items, or
                      general questions. The assistant routes each turn through
                      the benchmark backend and returns structured results.
                    </p>
                  </div>
                )}

                <ScrollToBottom className="absolute bottom-full left-1/2 -translate-x-1/2 mb-4 animate-in fade-in-0 zoom-in-95" />

                <div className="bg-muted rounded-2xl border shadow-xs mx-auto mb-8 w-full max-w-3xl relative z-10">
                  <form
                    onSubmit={handleSubmit}
                    className="grid grid-rows-[1fr_auto] gap-2 max-w-3xl mx-auto"
                  >
                    <textarea
                      data-testid="chat-input"
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

                    <div className="flex flex-col gap-3 p-3 pt-2">
                      <div className="flex flex-wrap items-end gap-3">
                        <label className="flex min-w-[210px] flex-col gap-1 text-xs text-gray-600">
                          <span className="pl-1 font-medium uppercase tracking-[0.14em] text-gray-500">
                            Paradigm
                          </span>
                          <select
                            data-testid="rec-model-select"
                            value={selectedRecModel}
                            onChange={(event) =>
                              setSelectedRecModel(
                                event.target.value as AppRecModel,
                              )
                            }
                            className="rounded-full border border-slate-300 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 shadow-sm outline-none transition focus:border-slate-400"
                          >
                            {REC_MODEL_OPTIONS.map((option) => (
                              <option key={option.value} value={option.value}>
                                {option.label}
                              </option>
                            ))}
                          </select>
                        </label>
                        <label className="flex min-w-[160px] flex-col gap-1 text-xs text-gray-600">
                          <span className="pl-1 font-medium uppercase tracking-[0.14em] text-gray-500">
                            Dataset
                          </span>
                          <select
                            data-testid="dataset-select"
                            value={selectedDataset}
                            onChange={(event) => {
                              hasManualDatasetSelection.current = true;
                              setSelectedDataset(event.target.value);
                            }}
                            className="rounded-full border border-slate-300 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 shadow-sm outline-none transition focus:border-slate-400"
                          >
                            {DATASET_OPTIONS.map((option) => (
                              <option key={option.value} value={option.value}>
                                {option.label}
                              </option>
                            ))}
                          </select>
                        </label>
                        <label className="flex min-w-[230px] flex-1 flex-col gap-1 text-xs text-gray-600">
                          <span className="pl-1 font-medium uppercase tracking-[0.14em] text-gray-500">
                            Dataset user
                          </span>
                          <select
                            data-testid="dataset-user-select"
                            value={selectedDatasetUserId}
                            onChange={(event) =>
                              setSelectedDatasetUsersByDataset((prev) => {
                                const next = { ...prev };
                                if (event.target.value) {
                                  next[selectedDataset] = event.target.value;
                                } else {
                                  delete next[selectedDataset];
                                }
                                return next;
                              })
                            }
                            className="rounded-full border border-slate-300 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 shadow-sm outline-none transition focus:border-slate-400"
                            disabled={datasetUsersLoading}
                          >
                            <option value="">
                              {datasetUsersLoading
                                ? "Loading profiles..."
                                : "Cold start"}
                            </option>
                            {datasetUsers.map((user) => (
                              <option key={user.user_id} value={user.user_id}>
                                {user.user_id} · {user.interaction_count} ints
                              </option>
                            ))}
                          </select>
                        </label>
                        {isLoading ? (
                          <Button
                            key="stop"
                            onClick={handleCancel}
                            type="button"
                            className="self-end"
                            data-testid="cancel-button"
                          >
                            <LoaderCircle className="w-4 h-4 animate-spin" />
                            Cancel
                          </Button>
                        ) : (
                          <Button
                            type="submit"
                            className="self-end transition-all shadow-md"
                            disabled={!input.trim()}
                            data-testid="send-button"
                          >
                            Send
                          </Button>
                        )}
                      </div>
                      <div
                        data-testid="benchmark-mode-copy"
                        className="grid gap-2 rounded-2xl border border-slate-200/90 bg-slate-50/90 px-4 py-3 text-xs leading-5 text-slate-600 sm:grid-cols-3"
                      >
                        <div>
                          <span className="block text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">
                            App user
                          </span>
                          <p className="mt-1">
                            Your signed-in account is{" "}
                            <span className="font-medium text-slate-800">
                              {stableUserId}
                            </span>
                            . It owns this chat thread, memory, and feedback.
                          </p>
                        </div>
                        <div>
                          <span className="block text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">
                            Benchmark mode
                          </span>
                          <p className="mt-1">
                            The <span className="font-medium text-slate-800">Dataset user</span>{" "}
                            selector does not switch your account. It injects an
                            offline-trained profile from{" "}
                            <span className="font-medium text-slate-800">
                              {selectedDatasetLabel}
                            </span>{" "}
                            for experimental non-cold-start checks.
                          </p>
                        </div>
                        <div>
                          <span className="block text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">
                            Active profile
                          </span>
                          <p className="mt-1">
                            {selectedDatasetUserSummary
                              ? `Using dataset user ${selectedDatasetUserSummary.user_id} (${selectedDatasetUserSummary.interaction_count} interactions).`
                              : `No dataset user selected for ${selectedDatasetLabel}; recommendations run in cold-start mode.`}
                          </p>
                        </div>
                      </div>
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
