import React, { useState } from 'react';
import { Box, Input, Image, Stack, Center } from '@chakra-ui/react';
import { MdInsertPhoto } from 'react-icons/md';
import { TextWIcon } from '../TextWIcon';

type ImageUploaderProps = {
  selectedImage: string | null, 
  onChange: (newSelectedImage: string) => void
}

function ImageUploader({selectedImage, onChange}: ImageUploaderProps): JSX.Element {

  const handleImageChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];

    if (file) {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = function () {
        onChange(reader.result as string);
      };
    }
  };

  return (
    <Stack w='full' p={3} border='1px solid var(--chakra-colors-gray-200)' background='white' borderRadius='2xl'>
      <TextWIcon icon={MdInsertPhoto} mb='8px'>Imagen</TextWIcon>
      {selectedImage && 
        <Center>
          <Image w='60%' aspectRatio='16 / 9' fit='scale-down' src={selectedImage} alt="Imagen seleccionada" />    
        </Center>
      }

      <Input border='none' type="file" accept="image/*" onChange={handleImageChange}/>
    </Stack>
  );
};

export { ImageUploader };
