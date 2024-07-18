'use client';

import { ROUTES } from "@/constants/router";
import { Button, Card, CardBody, CardHeader, Center, Heading, Text, VStack } from "@chakra-ui/react";
import { useRouter } from "next/navigation";

export default function Home() {
  const router = useRouter();

  const navigate = (route: ROUTES) => {
    router.push(route);
  }

  return (
    <>
      <Center h="100vh">
        <Card p={8} w="100%" maxW="600px" borderRadius='xl' style={{ "boxShadow": "#1db954 0px 8px 24px" }}>
          <CardHeader>
            <Heading as="h1" size="xl" textAlign="center">
              Proyecto Final - BDII 🗃️
            </Heading>
          </CardHeader>
          <CardBody>
            <VStack align='flex-start' spacing={4}>
              <Text fontSize="lg" textAlign="center">
                Selecciona a que sección deseas ir:
              </Text>

              <Button w ='full' onClick={() => navigate(ROUTES.SONGS)} colorScheme="green" size="lg">
                Índice Invertido Textual - Spotify
              </Button>
              <Button w ='full' onClick={() => navigate(ROUTES.IMAGES)} colorScheme="green" size="lg">
                Índice Multidimensional - Fashion Product Images
              </Button>
            </VStack>
          </CardBody>
        </Card>
      </Center>
    </>
  );
}