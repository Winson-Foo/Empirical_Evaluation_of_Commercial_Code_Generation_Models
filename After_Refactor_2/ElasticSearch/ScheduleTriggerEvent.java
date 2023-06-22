package org.elasticsearch.xpack.watcher.trigger.schedule;

import org.elasticsearch.ElasticsearchParseException;
import org.elasticsearch.xcontent.ParseField;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xcontent.XContentParser;
import org.elasticsearch.xpack.core.watcher.support.WatcherDateTimeUtils;
import org.elasticsearch.xpack.core.watcher.trigger.TriggerEvent;

import java.io.IOException;
import java.time.Clock;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;

public class ScheduleTriggerEvent extends TriggerEvent {

    private final ZonedDateTime scheduledTime;

    public ScheduleTriggerEvent(ZonedDateTime triggeredTime, ZonedDateTime scheduledTime) {
        this(null, triggeredTime, scheduledTime);
    }

    public ScheduleTriggerEvent(String jobName, ZonedDateTime triggeredTime, ZonedDateTime scheduledTime) {
        super(jobName, triggeredTime);
        this.scheduledTime = scheduledTime;
        data.put(Field.SCHEDULED_TIME.getPreferredName(), ZonedDateTime.ofInstant(scheduledTime.toInstant(), ZoneOffset.UTC));
    }

    public ZonedDateTime scheduledTime() {
        return scheduledTime;
    }

    @Override
    public String type() {
        return ScheduleTrigger.TYPE;
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        writeDateTime(Field.TRIGGERED_TIME.getPreferredName(), builder, triggeredTime);
        writeDateTime(Field.SCHEDULED_TIME.getPreferredName(), builder, scheduledTime);
        builder.endObject();
        return builder;
    }

    @Override
    public void recordDataXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject(ScheduleTrigger.TYPE);
        writeDateTime(Field.SCHEDULED_TIME.getPreferredName(), builder, scheduledTime);
        builder.endObject();
    }

    private void writeDateTime(String fieldName, XContentBuilder builder, ZonedDateTime dateTime) throws IOException {
        WatcherDateTimeUtils.writeDate(fieldName, builder, dateTime);
    }

    public static ScheduleTriggerEvent parse(XContentParser parser, String watchId, String context, Clock clock) throws IOException {
        ZonedDateTime triggeredTime = null;
        ZonedDateTime scheduledTime = null;
        String currentFieldName = null;

        XContentParser.Token token;
        while ((token = parser.nextToken()) != XContentParser.Token.END_OBJECT) {
            if (token == XContentParser.Token.FIELD_NAME) {
                currentFieldName = parser.currentName();
            } else if (Field.TRIGGERED_TIME.match(currentFieldName, parser.getDeprecationHandler())) {
                triggeredTime = parseDateMath(currentFieldName, parser, ZoneOffset.UTC, clock);
            } else if (Field.SCHEDULED_TIME.match(currentFieldName, parser.getDeprecationHandler())) {
                scheduledTime = parseDateMath(currentFieldName, parser, ZoneOffset.UTC, clock);
            } else {
                throw new ElasticsearchParseException(
                    "could not parse trigger event for [{}] for watch [{}]. unexpected token [{}]",
                    context,
                    watchId,
                    token
                );
            }
        }

        // should never be, it's fully controlled internally (not coming from the user)
        assert triggeredTime != null && scheduledTime != null;
        return new ScheduleTriggerEvent(triggeredTime, scheduledTime);
    }

    private static ZonedDateTime parseDateMath(String fieldName, XContentParser parser, ZoneOffset zoneOffset, Clock clock) throws IOException {
        try {
            return WatcherDateTimeUtils.parseDateMath(fieldName, parser, zoneOffset, clock);
        } catch (ElasticsearchParseException pe) {
            throw new ElasticsearchParseException(
                "could not parse [{}] trigger event for [{}] for watch [{}]. failed to parse date field [{}]",
                pe,
                ScheduleTriggerEngine.TYPE,
                context,
                watchId,
                fieldName
            );
        }
    }

    interface Field extends TriggerEvent.Field {
        ParseField SCHEDULED_TIME = new ParseField("scheduled_time");
    }
}