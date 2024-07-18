import { Box, VStack } from "@chakra-ui/react";
import { Image } from '@chakra-ui/react'

export function NavBar({image, children} : {image: string, children: React.ReactNode}) {
  return (
    <nav>
      <Box 
        w="100%" 
        h="240px"
        bgGradient="linear(primaryColor 0%, darkColor 90%, backgroundColor 100%)"
      ></Box>

      <VStack px={4} mt="-133.5px">
        <Image src={image} alt='Logo' height={20} />
        <Box mt={8} w="100%" maxW="600px">
          {children}
        </Box>
      </VStack>
    </nav>
  );
}
