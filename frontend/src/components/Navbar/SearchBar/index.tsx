"use client"

import { SongsContext } from "@/context/SongsContext";
import { Box, HStack, IconButton, Input, InputGroup, InputRightElement } from "@chakra-ui/react";
import React, { useContext } from "react";
import { FaSearch } from "react-icons/fa";

export function SearchBar() {
  const { search } = useContext(SongsContext);

  const [value, setValue] = React.useState('');

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const result = await search(value, 10, 'spanish', false);
  }
  return (
    <Box as="form" width='full' onSubmit={handleSubmit}>
        <HStack spacing="4">
          <InputGroup size="lg">
            <Input
              type="text"
              placeholder="Buscar canción"
              value={value}
              textColor='darkColor1'
              onChange={(e) => setValue(e.target.value)}
              borderRadius="full"
              variant='outline'
              bgColor='white'
              focusBorderColor='primaryColor'
              boxShadow="lg"
              pl={8}
              pr={16}
            />
            <InputRightElement>
              <IconButton
                type="submit"
                aria-label="Buscar"
                icon={<FaSearch />}
                colorScheme="green"
                borderRadius="full"
                right='0'
                bg="primaryColor"
                _hover={{ bg: 'primaryColor' }}
                _active={{ bg: 'primaryColor' }}
              />
            </InputRightElement>
          </InputGroup>
        </HStack>
      </Box>
  );
}