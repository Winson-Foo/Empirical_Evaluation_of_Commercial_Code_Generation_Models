// ViewButton.js
import Link from 'next/link';
import { Button, Text, Icon, Icons } from 'react-basics';
import useMessages from 'hooks/useMessages';
import useConfig from 'hooks/useConfig';

function ViewButton({ id, label }) {
  const { formatMessage } = useMessages();
  const { openExternal } = useConfig();

  return (
    <Link href={`/websites/${id}`} target={openExternal ? '_blank' : null}>
      <Button>
        <Icon>
          <Icons.External />
        </Icon>
        <Text>{formatMessage(label)}</Text>
      </Button>
    </Link>
  );
}

export default ViewButton;