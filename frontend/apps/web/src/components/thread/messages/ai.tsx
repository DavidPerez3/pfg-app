import { MarkdownText } from "../markdown-text";
import { AppMessage } from "@/lib/chat-types";
import { buildCopyPayload } from "@/lib/copy-payload";
import { CommandBar } from "./shared";
import { RecommendationCards } from "./recommendation-cards";
import { Button } from "@/components/ui/button";
import { LoaderCircle, ThumbsDown, ThumbsUp } from "lucide-react";

export function AssistantMessage({
  message,
  isLoading,
  onFeedback,
  onFollowUpPrompt,
  feedbackState = "idle",
}: {
  message: AppMessage;
  isLoading: boolean;
  onFeedback?: (rating: number) => void;
  onFollowUpPrompt?: (prompt: string) => void;
  feedbackState?: "idle" | "submitting" | "submitted";
}) {
  return (
    <div className="flex items-start mr-auto gap-2 group max-w-4xl">
      <div className="flex flex-col gap-2">
        <div className="py-1">
          <MarkdownText>{message.content}</MarkdownText>
        </div>
        {message.result ? (
          <RecommendationCards
            result={message.result}
            onFollowUpPrompt={onFollowUpPrompt}
          />
        ) : null}
        {onFeedback ? (
          <div className="flex flex-wrap items-center gap-2 pt-1">
            <span className="text-xs font-medium uppercase tracking-[0.14em] text-slate-500">
              Feedback
            </span>
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="rounded-full"
              disabled={feedbackState !== "idle"}
              onClick={() => onFeedback(5)}
            >
              {feedbackState === "submitting" ? (
                <LoaderCircle className="mr-1 h-3.5 w-3.5 animate-spin" />
              ) : (
                <ThumbsUp className="mr-1 h-3.5 w-3.5" />
              )}
              Useful
            </Button>
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="rounded-full"
              disabled={feedbackState !== "idle"}
              onClick={() => onFeedback(2)}
            >
              <ThumbsDown className="mr-1 h-3.5 w-3.5" />
              Needs work
            </Button>
            {feedbackState === "submitted" ? (
              <span className="text-xs text-emerald-700">
                Feedback stored for this answer.
              </span>
            ) : null}
          </div>
        ) : null}
        <div className="flex gap-2 items-center mr-auto opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100">
          <CommandBar
            content={buildCopyPayload(message)}
            isAiMessage={true}
            isLoading={isLoading}
          />
        </div>
      </div>
    </div>
  );
}

export function AssistantMessageLoading() {
  return (
    <div className="flex items-start mr-auto gap-2">
      <div className="flex items-center gap-1 rounded-2xl bg-muted px-4 py-2 h-8">
        <div className="w-1.5 h-1.5 rounded-full bg-foreground/50 animate-[pulse_1.5s_ease-in-out_infinite]" />
        <div className="w-1.5 h-1.5 rounded-full bg-foreground/50 animate-[pulse_1.5s_ease-in-out_0.5s_infinite]" />
        <div className="w-1.5 h-1.5 rounded-full bg-foreground/50 animate-[pulse_1.5s_ease-in-out_1s_infinite]" />
      </div>
    </div>
  );
}
