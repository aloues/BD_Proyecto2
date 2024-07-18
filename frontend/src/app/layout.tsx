import { Metadata } from "next";
import { fonts } from "./fonts";
import { Providers } from "./providers";
import { Box } from "@chakra-ui/react";

export const metadata: Metadata = {
  title: "Proyecto Final - BDII"
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={fonts.nunitoSans.variable}>
        <Providers>
          <Box minW='100vw' minH='100vh' w='100%' h='100%' backgroundColor='backgroundColor'>
            {children}
          </Box>
        </Providers>
      </body>
    </html>
  )
}
