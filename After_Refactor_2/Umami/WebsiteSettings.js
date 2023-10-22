//imports
import { useEffect, useState } from 'react';
import { Icons, Item, Tabs, Text, Button } from 'react-basics';
import { useRouter } from 'next/router';
import Link from 'next/link';

//components
import Page from 'components/layout/Page';
import Breadcrumb from 'components/common/Breadcrumb';
import PageHeader from 'components/common/PageHeader';
import WebsiteEditForm from 'components/pages/settings/websites/WebsiteEditForm';
import WebsiteData from 'components/pages/settings/websites/WebsiteData';
import TrackingCode from 'components/pages/settings/websites/TrackingCode';
import ShareUrl from 'components/pages/settings/websites/ShareUrl';

//hooks
import useApi from 'hooks/useApi';
import useMessages from 'hooks/useMessages';
import useConfig from 'hooks/useConfig';
import useToast from 'hooks/useToast';

function WebsiteSettings({ websiteId }) {

  const router = useRouter();
  const [values, setValues] = useState(null);
  const [tab, setTab] = useState('details');
 
  const { formatMessage, labels, messages } = useMessages();
  const { openExternal } = useConfig();
  const { get, useQuery } = useApi();

  //get website data
  const { data, isLoading, error } = useQuery(
    ['website', websiteId],
    () => get(`/websites/${websiteId}`),
    { enabled: !!websiteId, cacheTime: 0 },
  );

  //show toast on save
  const { toast, showToast } = useToast();
  const showSuccess = () => {
    showToast({ message: formatMessage(messages.saved), variant: 'success' });
  };

  //handle form submit
  const handleSave = data => {
    showSuccess();
    setValues(state => ({ ...state, ...data }));
  };

  //handle reset
  const handleReset = async value => {
    if (value === 'delete') {
      await router.push('/settings/websites');
    } else if (value === 'reset') {
      showSuccess();
    }
  };

  useEffect(() => {
    if (data) {
      setValues(data);
    }
  }, [data]);

  return (
    <Page loading={isLoading || error}>
      
      {toast}

      <PageHeader
        title={
          <Breadcrumb>
            <Item>
              <Link href="/settings/websites">{formatMessage(labels.websites)}</Link>
            </Item>
            <Item>{values?.name}</Item>
          </Breadcrumb>
        }
      >
        <Link href={`/websites/${websiteId}`} target={openExternal ? '_blank' : null}>
          <Button variant="primary">
            <Icons.External />
            <Text>{formatMessage(labels.view)}</Text>
          </Button>
        </Link>
      </PageHeader>

      <Tabs selectedKey={tab} onSelect={setTab} style={{ marginBottom: 30 }}>
        <Item key="details">{formatMessage(labels.details)}</Item>
        <Item key="tracking">{formatMessage(labels.trackingCode)}</Item>
        <Item key="share">{formatMessage(labels.shareUrl)}</Item>
        <Item key="data">{formatMessage(labels.data)}</Item>
      </Tabs>

      {tab === 'details' && (
        <WebsiteEditForm websiteId={websiteId} data={values} onSave={handleSave} />
      )}

      {tab === 'tracking' && <TrackingCode websiteId={websiteId} data={values} />}

      {tab === 'share' && <ShareUrl websiteId={websiteId} data={values} onSave={handleSave} />}
      
      {tab === 'data' && <WebsiteData websiteId={websiteId} onSave={handleReset} />}

    </Page>
  );
}

export default WebsiteSettings;

