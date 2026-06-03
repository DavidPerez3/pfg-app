import { v4 as uuidv4 } from "uuid";

import { AppMessage, AppStructuredResult } from "./chat-types";

export interface BackendChatMessage {
  role: string;
  content: string;
}

export function makeThreadTitle(text: string) {
  const clean = text.trim().replace(/\s+/g, " ");
  return clean.length <= 64 ? clean : `${clean.slice(0, 61)}...`;
}

export function normalizeBackendMessages(
  messages: BackendChatMessage[],
  fallbackTraceId?: string,
): AppMessage[] {
  return messages
    .filter((message) => message.role === "user" || message.role === "assistant")
    .map((message) => ({
      id: uuidv4(),
      role: message.role as "user" | "assistant",
      content: message.content,
      createdAt: new Date().toISOString(),
      traceId: fallbackTraceId,
    }));
}

export function sameBackendMessageShape(
  left: Pick<AppMessage, "role" | "content">,
  right: Pick<AppMessage, "role" | "content">,
): boolean {
  return left.role === right.role && left.content === right.content;
}

export function mergeBackendMessages(
  existingMessages: AppMessage[],
  backendMessages: BackendChatMessage[],
  fallbackTraceId?: string,
): AppMessage[] {
  const normalizedBackend = normalizeBackendMessages(
    backendMessages,
    fallbackTraceId,
  );
  if (!normalizedBackend.length) {
    return existingMessages;
  }

  const merged: AppMessage[] = [];
  let backendIndex = 0;

  while (
    backendIndex < normalizedBackend.length &&
    backendIndex < existingMessages.length &&
    sameBackendMessageShape(existingMessages[backendIndex], normalizedBackend[backendIndex])
  ) {
    merged.push(existingMessages[backendIndex]);
    backendIndex += 1;
  }

  for (; backendIndex < normalizedBackend.length; backendIndex += 1) {
    merged.push(normalizedBackend[backendIndex]);
  }

  return merged;
}

export function attachResultToLatestAssistant(
  messages: AppMessage[],
  result?: AppStructuredResult,
): AppMessage[] {
  if (!result) return messages;

  const nextMessages = [...messages];
  for (let index = nextMessages.length - 1; index >= 0; index -= 1) {
    if (nextMessages[index]?.role !== "assistant") continue;
    nextMessages[index] = {
      ...nextMessages[index],
      result,
    };
    break;
  }
  return nextMessages;
}
