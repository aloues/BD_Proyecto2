"use client"

import { Song } from "@/api/search/search.model";
import { Grid, HStack, Icon, IconButton, Text, VStack } from "@chakra-ui/react";
import Image from "next/image";
import { useRef, useState } from "react";
import { FaPause, FaPlay } from "react-icons/fa";
import { set } from "zod";

export function SongCard({ song }: { song: Song }) {
  const audioRef = useRef<HTMLAudioElement>(null);

  const [isPlaying, setIsPlaying] = useState(false);

  const togglePlay = () => {
    if (audioRef.current) {
      if (audioRef.current.paused) {
        audioRef.current.currentTime = 0;
        audioRef.current.play();
        setIsPlaying(true);
      } else {
        audioRef.current.pause();
        setIsPlaying(false);
      }
    }
  }

  return (
    <Grid w='100%' height='80px' templateColumns="80px 1fr 40px" gap={4} justifyItems='center' alignItems='center' >
      <Image src={song.album_image} alt={song.name} width={80} height={80} />
      <VStack w='100%' align="flex-start" justify="center">
        <Text fontWeight="bold" fontSize="lg" color="textColor" noOfLines={1}>
          {song.name}
        </Text>
        <Text fontSize="sm" color="textColor" noOfLines={1}>
          {song.artists.join(', ')}
        </Text>
      </VStack>
      
      { song.preview_url && <>
        <Icon 
        as={isPlaying ? FaPause : FaPlay}
        w={6}
        h={6}
        color="white"
        _hover={{ color: 'primaryColor', transform: 'scale(1.2)', cursor: 'pointer'}}
        _active={{ color: 'primaryColor' }}
        onClick={togglePlay}
        />

        <audio ref={audioRef} src={song.preview_url} />
      </>}
    </Grid>
  );
}