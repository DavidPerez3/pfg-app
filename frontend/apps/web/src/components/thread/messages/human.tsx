import { AppMessage } from "@/lib/chat-types";
import { CommandBar } from "./shared";

export function HumanMessage({
  message,
  isLoading,
}: {
  message: AppMessage;
  isLoading: boolean;
}) {
  return (
    <div className="flex items-center ml-auto gap-2 group max-w-xl">
      <div className="flex flex-col gap-2 ml-auto">
        <p className="px-4 py-2 rounded-3xl bg-muted w-fit ml-auto whitespace-pre-wrap">
          {message.content}
        </p>
        <div className="flex gap-2 items-center ml-auto opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100">
          <CommandBar
            content={message.content}
            isHumanMessage={true}
            isLoading={isLoading}
          />
        </div>
      </div>
    </div>
  );
}
