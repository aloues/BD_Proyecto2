"use client";

import { NavBar } from "@/components/Navbar";
import { SongsSearchBar } from "@/components/Navbar/SongsSearchBar";
import { SongCard } from "@/components/SongCard";
import { SongsContext, SongsProvider } from "@/context/SongsContext";
import { Spinner, Text, VStack } from "@chakra-ui/react";
import { useContext } from "react";

function SongsUI() {
  const { result, loading } = useContext(SongsContext);
  const { songs, executionTime } = result;

  return (
    <>
      <NavBar image="/spotify-white-logo.png">
        <SongsSearchBar />
      </NavBar>
      <main>
        <VStack w="100%" maxW="600px" mt={8} mx="auto" px={4}>
          <VStack w="100%" align="flex-start" pb={16}>
            <Text fontSize="2xl" fontWeight="bold" color="textColor">
              Top Songs
            </Text>
            <Text fontSize="lg" fontWeight="bold" color="textColor">
              {loading
                ? "Cargando..."
                : songs.length === 0
                ? "No se encontraron canciones"
                : `Se encontraron ${songs.length} canciones (${
                    Math.round(executionTime * 100) / 100
                  }ms)`}
            </Text>

            {loading ? (
              <Spinner size="xl" color="primaryColor" />
            ) : (
              songs.map((song) => <SongCard key={song.id} song={song} />)
            )}
          </VStack>
        </VStack>
      </main>
    </>
  );
}

export default function Songs() {
  return (
    <SongsProvider>
      <SongsUI />
    </SongsProvider>
  );
}
