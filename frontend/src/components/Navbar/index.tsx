import { Box, VStack } from "@chakra-ui/react";
import { SearchBar } from "./SearchBar";
import Image from "next/image";

export function NavBar() {
  return (
    <nav>
      <Box 
        w="100%" 
        h="240px"
        bgGradient="linear(primaryColor 0%, darkColor 90%, backgroundColor 100%)"
      ></Box>

      <VStack px={4} mt="-133.5px">
        <Image src='/spotify-white-logo.png' alt='Spotify Logo' width={212.5} height={63.5} />
        <Box mt={8} w="100%" maxW="600px">
          <SearchBar />
        </Box>
      </VStack>
    </nav>
  );
}
