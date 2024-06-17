"use client"

import { searchSong } from "@/api/search";
import { SearchResult, Song } from "@/api/search/search.model";
import { useToast } from "@chakra-ui/react";
import { createContext, useState } from "react";


interface SongsContextProps {
  search: (query: string, k: number, language: string, use_postgres: boolean) => Promise<SearchResult>;
  loading: boolean;
  result: SearchResult;
}

const SongsContext = createContext<SongsContextProps>({
  search: async () => Promise.resolve({ songs: [], executionTime: 0 }),
  loading: false,
  result: { songs: [], executionTime: 0 },
});

function SongsProvider({ children }: { children: React.ReactNode }) {
  const toast = useToast();

  const errorToast = (title: string, description: string) => {
    toast({
      title,
      description,
      status: 'error',
      duration: 3000,
      isClosable: true,
    });
  }

  const [result, setResult] = useState<SearchResult>({ songs: [], executionTime: 0 });
  const [loading, setLoading] = useState(false);
  const search = async (query: string, k: number, language: string, use_postgres: boolean) => {
    setLoading(true);
    try {
      const result = await searchSong(query, k, language, use_postgres);
      setResult(result);
      setLoading(false);
      return result;
    } catch (error) {
      errorToast('Error', 'No se ha podido realizar la búsqueda');
      setLoading(false);
      return { songs: [], executionTime: 0};
    }
  }

  return (
    <SongsContext.Provider value={{ search, result, loading }}>
      {children}
    </SongsContext.Provider>
  );
}

export { SongsContext, SongsProvider };