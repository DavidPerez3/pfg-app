import { Copy, CopyCheck } from "lucide-react";
import { useState } from "react";
import { TooltipIconButton } from "../tooltip-icon-button";

function ContentCopyable({
  content,
  disabled,
}: {
  content: string;
  disabled: boolean;
}) {
  const [copied, setCopied] = useState(false);

  const handleCopy = (e: React.MouseEvent<HTMLButtonElement>) => {
    e.stopPropagation();
    navigator.clipboard.writeText(content);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 2000);
  };

  return (
    <TooltipIconButton
      onClick={handleCopy}
      variant="ghost"
      tooltip="Copy content"
      disabled={disabled}
    >
      {copied ? <CopyCheck className="text-green-500" /> : <Copy />}
    </TooltipIconButton>
  );
}

export function CommandBar({
  content,
  isLoading,
}: {
  content: string;
  isLoading: boolean;
  isHumanMessage?: boolean;
  isAiMessage?: boolean;
}) {
  return (
    <div className="flex items-center gap-2">
      <ContentCopyable content={content} disabled={isLoading} />
    </div>
  );
}

export function BranchSwitcher() {
  return null;
}
