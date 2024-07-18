import React, { ReactNode } from 'react';
import { HStack, Icon, Text, StackProps } from '@chakra-ui/react';
import { IconType } from 'react-icons';

interface TextWithIconProps extends StackProps {
  icon: IconType;
  children: ReactNode;
}

function TextWIcon({ icon, children, ...rest }: TextWithIconProps): JSX.Element {
  return (
    <HStack align="center" {...rest}>
      <Icon as={icon} color='primaryColor' />
      <Text>{children}</Text>
    </HStack>
  );
};

export { TextWIcon };