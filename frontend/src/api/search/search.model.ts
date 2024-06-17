import { z } from "zod";

export const SongSchema = z.object({
  name: z.string(),
  artists: z.array(z.string()),
  album: z.string(),
  preview_url: z.string(),
  album_image: z.string(),
  id: z.string(),
  score: z.number(),
});

export type Song = z.infer<typeof SongSchema>;

export const SearchResultSchema = z.object({
  songs: z.array(SongSchema),
  executionTime: z.number(),
});

export type SearchResult = z.infer<typeof SearchResultSchema>;

