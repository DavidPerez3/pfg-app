import { AppRecommendedItem, AppStructuredResult } from "@/lib/chat-types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

function splitGenres(genres?: string): string[] {
  if (!genres) return [];
  return genres
    .split("|")
    .map((genre) => genre.trim())
    .filter(Boolean)
    .slice(0, 4);
}

function scoreLabel(item: AppRecommendedItem, index: number) {
  return `Rank #${index + 1}`;
}

export function RecommendationCards({
  result,
  onFollowUpPrompt,
}: {
  result: AppStructuredResult;
  onFollowUpPrompt?: (prompt: string) => void;
}) {
  if (!result.items.length) return null;

  return (
    <div data-testid="structured-result" className="mt-4 flex w-full flex-col gap-3">
      <div className="flex flex-col gap-1">
        <h3 className="text-sm font-semibold uppercase tracking-[0.18em] text-slate-500">
          {result.title}
        </h3>
        {result.subtitle ? (
          <p className="text-sm text-slate-600">{result.subtitle}</p>
        ) : null}
        {result.cold_start ? (
          <div className="pt-1">
            <span className="rounded-full border border-sky-200 bg-sky-50 px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.14em] text-sky-700">
              Cold start
            </span>
          </div>
        ) : null}
        {result.explanation ? (
          <p className="text-sm leading-6 text-slate-700">
            {result.explanation}
          </p>
        ) : null}
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        {result.items.map((item, index) => {
          const genres = splitGenres(item.genres);
          return (
            <Card
              key={`${result.kind}-${item.title}-${index}`}
              data-testid="result-item-card"
              className="gap-4 border-slate-200 bg-white/95 py-4 shadow-[0_16px_40px_-28px_rgba(15,23,42,0.45)]"
            >
              <CardHeader className="gap-3 px-4">
                <div className="flex items-start justify-between gap-4">
                  <CardTitle
                    data-testid="result-item-title"
                    className="text-base leading-6 text-slate-900"
                  >
                    {item.title}
                  </CardTitle>
                  <span className="rounded-full border border-amber-300 bg-amber-50 px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.14em] text-amber-700">
                    {scoreLabel(item, index)}
                  </span>
                </div>
              </CardHeader>
              <CardContent className="flex flex-col gap-3 px-4">
                {genres.length > 0 ? (
                  <div className="flex flex-wrap gap-2">
                    {genres.map((genre) => (
                      <span
                        key={`${item.title}-${genre}`}
                        className="rounded-full bg-slate-100 px-2.5 py-1 text-xs font-medium text-slate-700"
                      >
                        {genre}
                      </span>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-slate-500">
                    No genre metadata available for this item.
                  </p>
                )}
              </CardContent>
            </Card>
          );
        })}
      </div>

      {result.follow_up_prompts?.length ? (
        <div className="flex flex-col gap-2 pt-1">
          <span className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">
            Refine this list
          </span>
          <div className="flex flex-wrap gap-2">
            {result.follow_up_prompts.map((prompt) => (
              <Button
                key={prompt}
                data-testid="follow-up-prompt-button"
                type="button"
                size="sm"
                variant="outline"
                className="rounded-full text-xs"
                onClick={() => onFollowUpPrompt?.(prompt)}
              >
                {prompt}
              </Button>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}
