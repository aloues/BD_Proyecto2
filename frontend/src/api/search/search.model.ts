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

export const SongsSearchResultSchema = z.object({
  songs: z.array(SongSchema),
  executionTime: z.number(),
});

export type SongsSearchResult = z.infer<typeof SongsSearchResultSchema>;

export const TopImageSchema = z.object({
  id: z.number(),
  name: z.string(),
  price: z.number(),
  url: z.string(),
  score: z.number(),
  variantName: z.string(),
  brandName: z.string(),
});

export type TopImage = z.infer<typeof TopImageSchema>;

export const ImagesSearchResultSchema = z.object({
  images: z.array(TopImageSchema),
  executionTime: z.number(),
});

export type ImagesSearchResult = z.infer<typeof ImagesSearchResultSchema>;



