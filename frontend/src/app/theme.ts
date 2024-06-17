import { background, extendTheme } from "@chakra-ui/react";

export const theme = extendTheme({
    html: {
      heading: 'var(--font-nunito-sans)',
      body: 'var(--nunito-sans)',
    },
    colors: {
      primaryColor: "#1db954",
      backgroundColor: "#121212",
      darkColor: "#212121",
      darkColor2: "#535353",
      lightColor: "#b3b3b3",
      textColor: "#ffffff",
    }
});