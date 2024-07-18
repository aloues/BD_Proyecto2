"use client"

import { SongsContext } from "@/context/SongsContext";
import { Box, HStack, IconButton, Input, InputGroup, InputRightElement, NumberDecrementStepper, NumberIncrementStepper, NumberInput, NumberInputField, NumberInputStepper, Select, VStack } from "@chakra-ui/react";
import React, { useContext } from "react";
import { FaSearch } from "react-icons/fa";

export function SongsSearchBar() {
  const { search } = useContext(SongsContext);

  const [value, setValue] = React.useState('');
  const [k, setK] = React.useState('10');
  const [lang, setLang] = React.useState('spanish');
  const [index, setIndex] = React.useState('own');

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const usePostgres = index === 'postgres';
    await search(value, parseInt(k), lang, usePostgres);
  }
  return (
    <Box as="form" width='full' onSubmit={handleSubmit}>
        <VStack spacing="4">
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
          <HStack spacing="4">
            <NumberInput
              height='full'
              onChange={(newK) => setK(newK)}
              value={k}
              min={1}
              bgColor='white'
              borderRadius="full"
              textColor='darkColor1'
              focusBorderColor='primaryColor'
            >
              <NumberInputField />
              <NumberInputStepper>
                <NumberIncrementStepper />
                <NumberDecrementStepper />
              </NumberInputStepper>
            </NumberInput>

            <Select value={lang} onChange={(e) => setLang(e.target.value)} bgColor='white' borderRadius="full" textColor='darkColor1' focusBorderColor='primaryColor'>
              <option value="spanish">Español</option>
              <option value="english">Inglés</option>
            </Select>

            <Select value={index} onChange={(e) => setIndex(e.target.value)} bgColor='white' borderRadius="full" textColor='darkColor1' focusBorderColor='primaryColor'>
              <option value="own">Índice propio</option>
              <option value="postgres">Índice Postgres</option>
            </Select>
          </HStack>
        </VStack>
      </Box>
  );
}