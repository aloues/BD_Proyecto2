import { MULTI_DIMENSIONAL_INDEX_MODELS } from "@/constants/router";
import { api } from "..";
import { SongsSearchResultSchema, ImagesSearchResultSchema } from "./search.model";

export async function searchSong(query: string, k: number, language: string, use_postgres: boolean) {
  const res = await api.post("/songs/search", { query, k, language, use_postgres });
  return SongsSearchResultSchema.parse(res.data);
}

export async function searchImage(query: string, k: number, n: number, model: MULTI_DIMENSIONAL_INDEX_MODELS) {
  const res = await api.post("/images/search", { query, k, n, model });
  return ImagesSearchResultSchema.parse(res.data);
}