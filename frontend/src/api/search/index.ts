import { api } from "..";
import { SearchResultSchema } from "./search.model";

export async function searchSong(query: string, k: number, language: string, use_postgres: boolean) {
  const res = await api.post("/search", { query, k, language, use_postgres });
  return SearchResultSchema.parse(res.data);
}