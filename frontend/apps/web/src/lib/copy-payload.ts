import { AppMessage } from "./chat-types";

const REC_MODEL_LABELS: Record<string, string> = {
  mf: "Matrix Factorization",
  matrix_factorization: "Matrix Factorization",
  two_tower: "Two-Tower",
  two_tower_wide_deep: "Two-Tower + Wide&Deep",
  sasrec: "SASRec",
  llm_rag: "LLM + RAG",
};

const DATASET_LABELS: Record<string, string> = {
  movielens: "MovieLens",
  lastfm: "LastFM",
  yelp: "Yelp",
  amazon_electronics: "Amazon Electronics",
  foursquare: "Foursquare",
};

export function prettyRecModel(recModel?: string): string {
  if (!recModel) return "Unknown";
  return REC_MODEL_LABELS[recModel] ?? recModel;
}

export function prettyDataset(dataset?: string): string {
  if (!dataset) return "Unknown";
  return DATASET_LABELS[dataset] ?? dataset;
}

export function cleanMessageSummary(messageContent: string): string {
  return messageContent
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .join("\n");
}

export function buildCopyPayload(message: AppMessage): string {
  const baseContent = cleanMessageSummary(message.content);
  const result = message.result;

  if (!result) {
    return baseContent;
  }

  const explanation = (result.explanation ?? "").trim() || baseContent;
  const sections: string[] = [];

  if (result.kind === "recommendations") {
    sections.push("Recommendation Result");
    sections.push(`Paradigm: ${prettyRecModel(result.rec_model)}`);
    sections.push(`Dataset: ${prettyDataset(result.dataset)}`);
    if (result.dataset_user_id) {
      sections.push(`Dataset user: ${result.dataset_user_id}`);
    }
    if (typeof result.cold_start === "boolean") {
      sections.push(`Mode: ${result.cold_start ? "Cold-start" : "Non-cold-start"}`);
    }
    if (result.trace_id) {
      sections.push(`Trace ID: ${result.trace_id}`);
    }
    if (explanation) {
      sections.push(`Explanation\n${explanation}`);
    }
    if (result.items.length > 0) {
      sections.push(
        `Top-${result.items.length} items\n${result.items
          .map((item, index) => {
            const parts = [`${index + 1}. ${item.title}`];
            parts.push(`rank: #${index + 1}`);
            if (item.genres?.trim()) {
              parts.push(`genres: ${item.genres.trim()}`);
            }
            return parts.join(" - ");
          })
          .join("\n")}`,
      );
    }
    return sections.join("\n\n");
  }

  const header =
    result.kind === "similar_items" ? "Similarity Result" : "Search Result";
  sections.push(header);
  if (result.dataset) {
    sections.push(`Dataset: ${prettyDataset(result.dataset)}`);
  }
  if (result.rec_model) {
    sections.push(`Paradigm: ${prettyRecModel(result.rec_model)}`);
  }
  if (result.seed_title) {
    sections.push(`Seed item: ${result.seed_title}`);
  }
  if (result.query) {
    sections.push(`Query: ${result.query}`);
  }
  if (result.trace_id) {
    sections.push(`Trace ID: ${result.trace_id}`);
  }
  if (explanation) {
    sections.push(`Summary\n${explanation}`);
  }
  if (result.items.length > 0) {
    sections.push(
      `Ranked items\n${result.items
        .map((item, index) => {
          const parts = [`${index + 1}. ${item.title}`];
          parts.push(`rank: #${index + 1}`);
          if (item.genres?.trim()) {
            parts.push(`genres: ${item.genres.trim()}`);
          }
          return parts.join(" - ");
        })
        .join("\n")}`,
    );
  }
  return sections.join("\n\n");
}
