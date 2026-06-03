export type AppMessageRole = "user" | "assistant";
export type AppResultKind =
  | "recommendations"
  | "search_results"
  | "similar_items";
export type AppRecModel =
  | "mf"
  | "two_tower"
  | "two_tower_wide_deep"
  | "sasrec"
  | "llm_rag";

export interface AppRecommendedItem {
  title: string;
  score: number;
  genres?: string;
}

export interface AppStructuredResult {
  kind: AppResultKind;
  title: string;
  subtitle?: string;
  items: AppRecommendedItem[];
  dataset?: string;
  rec_model?: string;
  dataset_user_id?: string;
  cold_start?: boolean | null;
  query?: string;
  seed_title?: string;
  trace_id?: string;
  explanation?: string | null;
  follow_up_prompts?: string[];
}

export interface AppMessage {
  id: string;
  role: AppMessageRole;
  content: string;
  createdAt: string;
  traceId?: string;
  result?: AppStructuredResult;
}

export interface AppThread {
  thread_id: string;
  title: string;
  updated_at: string;
  messages: AppMessage[];
}

export interface DatasetUserOption {
  user_id: string;
  interaction_count: number;
}

export interface DatasetUsersResponse {
  dataset: string;
  users: DatasetUserOption[];
  total_available: number;
}
