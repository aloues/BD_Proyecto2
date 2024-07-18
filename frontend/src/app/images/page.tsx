"use client"

import { ImageCard } from "@/components/ImageCard";
import { NavBar } from "@/components/Navbar";
import { ImagesSearchBar } from "@/components/Navbar/ImagesSearchBar";
import { SongCard } from "@/components/SongCard";
import { ImagesContext, ImagesProvider } from "@/context/ImagesContext";
import { Box, Spinner, Text, VStack } from "@chakra-ui/react";
import { useContext } from "react";

function ImagesUI() {
  const { result, loading } = useContext(ImagesContext);
  const { images, executionTime } = result;
  
  return (
    <>
      <NavBar image="./fashion-white-logo.png" >
        <ImagesSearchBar />
      </NavBar>
      <main>
        <VStack w='100%' maxW='600px' mt={8} mx='auto' px={4}>
          <VStack w='100%' align='flex-start' pb={16}>
            <Text fontSize="2xl" fontWeight="bold" color='textColor'>
              Top Images
            </Text>
            <Text fontSize="lg" fontWeight="bold" color='textColor'>
              {
                loading ? 'Cargando...' :
                `Se encontraron ${images.length} imágenes (${Math.round(executionTime * 100) / 100}s)`
              }
            </Text>

            {
              loading ? <Spinner size="xl" color="primaryColor" /> :
              images.map((image) => (
                <ImageCard key={image.id} image={image} />
              ))
            }
          </VStack>
        </VStack>
      </main>
    </>
  );
}

export default function Images(){
  return (
    <ImagesProvider>
      <ImagesUI />
    </ImagesProvider>
  );
}