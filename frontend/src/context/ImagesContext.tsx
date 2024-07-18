"use client"

import { searchImage, searchSong } from "@/api/search";
import { ImagesSearchResult, TopImage } from "@/api/search/search.model";
import { MULTI_DIMENSIONAL_INDEX_MODELS } from "@/constants/router";
import { useToast } from "@chakra-ui/react";
import { createContext, useState } from "react";


interface ImagesContextProps {
  search: (query: string, k: number, n: number, model: MULTI_DIMENSIONAL_INDEX_MODELS) => Promise<ImagesSearchResult>;
  loading: boolean;
  result: ImagesSearchResult;
}

const ImagesContext = createContext<ImagesContextProps>({
  search: async () => Promise.resolve({ images: [], executionTime: 0 }),
  loading: false,
  result: { images: [], executionTime: 0 },
});

function ImagesProvider({ children }: { children: React.ReactNode }) {
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

  const [result, setResult] = useState<ImagesSearchResult>({ images: [], executionTime: 0 });
  const [loading, setLoading] = useState(false);
  const search = async (query: string, k: number, n: number, model: MULTI_DIMENSIONAL_INDEX_MODELS) => {
    setLoading(true);
    try {
      const result = await searchImage(query, k, n, model);
      setResult(result);
      setLoading(false);
      return result;
    } catch (error) {
      errorToast('Error', 'No se ha podido realizar la búsqueda');
      setLoading(false);
      return { images: [], executionTime: 0};
    }
  }

  return (
    <ImagesContext.Provider value={{ search, result, loading }}>
      {children}
    </ImagesContext.Provider>
  );
}

export { ImagesContext, ImagesProvider };