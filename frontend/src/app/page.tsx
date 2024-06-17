"use client"

import { NavBar } from "@/components/Navbar";
import { SongCard } from "@/components/SongCard";
import { SongsContext } from "@/context/SongsContext";
import { Spinner, Text, VStack } from "@chakra-ui/react";
import { useContext } from "react";

export default function Home() {
  const { result, loading } = useContext(SongsContext);
  const { songs, executionTime } = result;

  return (
    <>
      <NavBar />
      <main>
        <VStack w='100%' maxW='600px' mt={8} mx='auto' px={4}>
          <VStack w='100%' align='flex-start' pb={16}>
            <Text fontSize="2xl" fontWeight="bold" color='textColor'>
              Top Songs
            </Text>
            <Text fontSize="lg" fontWeight="bold" color='textColor'>
              {
                loading ? 'Cargando...' :
                songs.length === 0 ? 'No se encontraron canciones' :
                `Se encontraron ${songs.length} canciones (${Math.round(executionTime)}ms)`
              }
            </Text>

            {
              loading ? <Spinner size="xl" color="primaryColor" /> :
              songs.map((song) => <SongCard key={song.id} song={song} />)
            }
          </VStack>
        </VStack>
      </main>
    </>
  );
}
