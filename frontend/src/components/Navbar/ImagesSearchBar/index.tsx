"use client"

import { ImageUploader } from "@/components/ImageUploader";
import { MULTI_DIMENSIONAL_INDEX_MODELS } from "@/constants/router";
import { ImagesContext } from "@/context/ImagesContext";
import { Box, HStack, IconButton, NumberDecrementStepper, NumberIncrementStepper, NumberInput, NumberInputField, NumberInputStepper, Select, VStack } from "@chakra-ui/react";
import React, { useContext } from "react";
import { FaSearch } from "react-icons/fa";

export function ImagesSearchBar() {
  const { search } = useContext(ImagesContext);

  const [value, setValue] = React.useState('');
  const [k, setK] = React.useState('8');
  const [model, setModel] = React.useState<MULTI_DIMENSIONAL_INDEX_MODELS>(MULTI_DIMENSIONAL_INDEX_MODELS.KNN_SEQUENTIAL);
  const [n, setN] = React.useState('32000');

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    await search(value, parseInt(k), parseInt(n), model);
  }

  const onImageChange = (image: string) => {
    setValue(image);
  }

  return (
    <Box as="form" width='full' onSubmit={handleSubmit}>
        <VStack spacing="4">
          <HStack w='100%'>
            <ImageUploader 
              selectedImage={value}
              onChange={onImageChange}
            />
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
          </HStack>

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

            <Select value={n} onChange={(e) => setN(e.target.value)} bgColor='white' borderRadius="full" textColor='darkColor1' focusBorderColor='primaryColor'>
              <option value="1000">1000</option>
              <option value="2000">2000</option>
              <option value="4000">4000</option>
              <option value="8000">8000</option>
              <option value="16000">16000</option>
              <option value="32000">32000</option>
              <option value="44000">44000</option>
            </Select>

            <Select value={model} onChange={(e) => setModel(e.target.value as MULTI_DIMENSIONAL_INDEX_MODELS)} bgColor='white' borderRadius="full" textColor='darkColor1' focusBorderColor='primaryColor'>
              <option value={MULTI_DIMENSIONAL_INDEX_MODELS.KNN_SEQUENTIAL}>KnnSequential (D = 128)</option>
              <option value={MULTI_DIMENSIONAL_INDEX_MODELS.KNN_R_TREE}>KnnRTree (D = 128)</option>
              <option value={MULTI_DIMENSIONAL_INDEX_MODELS.HIGH_D_32}>HighD32 (D = 32)</option>
              <option value={MULTI_DIMENSIONAL_INDEX_MODELS.HIGH_D_16}>HighD16 (D = 16)</option>
              <option value={MULTI_DIMENSIONAL_INDEX_MODELS.HIGH_D_8}>HighD8 (D = 8)</option>
            </Select>
          </HStack>
        </VStack>
      </Box>
  );
}