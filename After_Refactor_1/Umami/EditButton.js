// EditButton.js
import Link from 'next/link';
import { Button, Text, Icon, Icons } from 'react-basics';
import useMessages from 'hooks/useMessages';

function EditButton({ id, label }) {
  const { formatMessage } = useMessages();

  return (
    <Link href={`/settings/websites/${id}`}>
      <Button>
        <Icon>
          <Icons.Edit />
        </Icon>
        <Text>{formatMessage(label)}</Text>
      </Button>
    </Link>
  );
}

export default EditButton;