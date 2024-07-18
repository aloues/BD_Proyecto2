"use client";

import { TopImage } from "@/api/search/search.model";
import { Badge, Box, Card, CardBody, CardFooter, Heading, Image, Stack, Text } from "@chakra-ui/react";

export function ImageCard({ image }: { image: TopImage }) {
  return (
    <Card
      direction={{ base: 'column', sm: 'row' }}
      overflow='hidden'
      variant='outline'
      w='full'
      size='md'
      maxH={{ base: 'auto', sm: '240px' }}
    >
      <Image
        objectFit='cover'
        maxW={{ base: '100%', sm: '180px' }}
        src={image.url}
        alt={image.name}
      />

      <Stack>
        <CardBody>
          <Heading size='md'>{image.name}</Heading>
          <Text color='gray.500'>
            {image.variantName} - {image.brandName}
          </Text>

          <Badge colorScheme='green'>
            Score: {Math.round(image.score * 100) / 100}
          </Badge>
        </CardBody>

        <CardFooter>
          <Text py='2' fontWeight='bold' fontSize='lg'>
            S/ {image.price / 10}
          </Text>
        </CardFooter>
      </Stack>
    </Card>
  );
}
