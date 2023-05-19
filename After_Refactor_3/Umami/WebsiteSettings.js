import { useEffect, useState } from 'react';
import { Breadcrumbs, Item, Tabs, useToast, Button, Text, Icon, Icons } from 'react-basics';
import { useRouter } from 'next/router';
import Link from 'next/link';
import Page from 'components/layout/Page';
import PageHeader from 'components/layout/PageHeader';
import WebsiteEditForm from 'components/pages/settings/websites/WebsiteEditForm';
import WebsiteData from 'components/pages/settings/websites/WebsiteData';
import TrackingCode from 'components/pages/settings/websites/TrackingCode';
import ShareUrl from 'components/pages/settings/websites/ShareUrl';
import { useApi } from 'hooks/useApi';
import { useMessages } from 'hooks/useMessages';
import { useConfig } from 'hooks/useConfig';

const WEBSITES_LABEL = 'websites';
const VIEW_LABEL = 'view';
const DETAILS_LABEL = 'details';
const TRACKING_CODE_LABEL = 'trackingCode';
const SHARE_URL_LABEL = 'shareUrl';
const DATA_LABEL = 'data';

export function WebsiteSettings({ websiteId }) {
  const router = useRouter();
  const { formatMessage, labels, messages } = useMessages();
  const { openExternal } = useConfig();
  const { get } = useApi();
  const { showToast } = useToast();
  const [websiteData, setWebsiteData] = useState(null);
  const [selectedTabKey, setSelectedTabKey] = useState(DETAILS_LABEL);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const data = await get(`/websites/${websiteId}`);
        setWebsiteData(data);
      } catch (error) {
        console.error(error);
      }
    };
    if (websiteId) {
      fetchData();
    }
  }, [get, websiteId]);

  const handleSave = (data) => {
    showToast({ message: formatMessage(messages.saved), variant: 'success' });
    setWebsiteData((prevState) => ({ ...prevState, ...data }));
  };

  const handleReset = async (value) => {
    try {
      if (value === 'delete') {
        await router.push('/settings/websites');
      } else if (value === 'reset') {
        showToast({ message: formatMessage(messages.saved), variant: 'success' });
      }
    } catch (error) {
      console.error(error);
    }
  };

  const websiteName = websiteData?.name ?? '';

  return (
    <Page loading={!websiteData}>
      {toast}
      <PageHeader
        title={
          <Breadcrumbs>
            <Item>
              <Link href="/settings/websites">{formatMessage(labels[WEBSITES_LABEL])}</Link>
            </Item>
            <Item>{websiteName}</Item>
          </Breadcrumbs>
        }
      >
        <Link href={`/websites/${websiteId}`} target={openExternal ? '_blank' : null}>
          <Button variant="primary">
            <Icon>
              <Icons.External />
            </Icon>
            <Text>{formatMessage(labels[VIEW_LABEL])}</Text>
          </Button>
        </Link>
      </PageHeader>
      <Tabs selectedKey={selectedTabKey} onSelect={setSelectedTabKey} style={{ marginBottom: 30 }}>
        <Item key={DETAILS_LABEL}>{formatMessage(labels[DETAILS_LABEL])}</Item>
        <Item key={TRACKING_CODE_LABEL}>{formatMessage(labels[TRACKING_CODE_LABEL])}</Item>
        <Item key={SHARE_URL_LABEL}>{formatMessage(labels[SHARE_URL_LABEL])}</Item>
        <Item key={DATA_LABEL}>{formatMessage(labels[DATA_LABEL])}</Item>
      </Tabs>
      {selectedTabKey === DETAILS_LABEL && (
        <WebsiteEditForm websiteId={websiteId} data={websiteData} onSave={handleSave} />
      )}
      {selectedTabKey === TRACKING_CODE_LABEL && (
        <TrackingCode websiteId={websiteId} data={websiteData} />
      )}
      {selectedTabKey === SHARE_URL_LABEL && (
        <ShareUrl websiteId={websiteId} data={websiteData} onSave={handleSave} />
      )}
      {selectedTabKey === DATA_LABEL && <WebsiteData websiteId={websiteId} onSave={handleReset} />}
    </Page>
  );
}

export default WebsiteSettings;

