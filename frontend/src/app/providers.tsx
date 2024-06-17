'use client'

import { ChakraProvider } from '@chakra-ui/react'
import { theme } from './theme';
import { SongsProvider } from '@/context/SongsContext';

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <SongsProvider>
      <ChakraProvider theme={theme}>
        {children}
      </ChakraProvider>
    </SongsProvider>
  );
}