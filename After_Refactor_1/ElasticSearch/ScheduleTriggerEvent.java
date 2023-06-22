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

    private static final String SCHEDULE_TRIGGER_TYPE = "schedule_trigger_type";
    private static final String UTC = "UTC";

    private final ZonedDateTime scheduledTime;

    public ScheduleTriggerEvent(ZonedDateTime triggeredTime, ZonedDateTime scheduledTime) {
        this(null, triggeredTime, scheduledTime);
    }

    public ScheduleTriggerEvent(String jobName, ZonedDateTime triggeredTime, ZonedDateTime scheduledTime) {
        super(jobName, triggeredTime);
        this.scheduledTime = scheduledTime;
        data.put(Field.SCHEDULED_TIME.getPreferredName(), ZonedDateTime.ofInstant(scheduledTime.toInstant(), ZoneOffset.UTC));
    }

    @Override
    public String type() {
        return ScheduleTrigger.TYPE;
    }

    public ZonedDateTime scheduledTime() {
        return scheduledTime;
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        WatcherDateTimeUtils.writeDate(Field.TRIGGERED_TIME.getPreferredName(), builder, triggeredTime);
        WatcherDateTimeUtils.writeDate(Field.SCHEDULED_TIME.getPreferredName(), builder, scheduledTime);
        return builder.endObject();
    }

    @Override
    public void recordDataXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject(SCHEDULE_TRIGGER_TYPE);
        WatcherDateTimeUtils.writeDate(Field.SCHEDULED_TIME.getPreferredName(), builder, scheduledTime);
        builder.endObject();
    }

    public static ScheduleTriggerEvent parse(XContentParser parser, String watchId, String context, Clock clock) throws IOException {
        ZonedDateTime triggeredTime = null;
        ZonedDateTime scheduledTime = null;

        try (XContentParser.Token token = parser.nextToken()) {
            while (token != XContentParser.Token.END_OBJECT) {
                if (token == XContentParser.Token.FIELD_NAME) {
                    String currentFieldName = parser.currentName();
                    token = parser.nextToken();
                    if (token == XContentParser.Token.VALUE_NULL) {
                        // skip null values
                        continue;
                    }

                    if (Field.TRIGGERED_TIME.match(currentFieldName, parser.getDeprecationHandler())) {
                        triggeredTime = parseDateTime(currentFieldName, parser, clock);
                    } else if (Field.SCHEDULED_TIME.match(currentFieldName, parser.getDeprecationHandler())) {
                        scheduledTime = parseDateTime(currentFieldName, parser, clock);
                    } else {
                        throw new ElasticsearchParseException(
                                "could not parse trigger event for [{}] for watch [{}]. unknown field [{}]",
                                context,
                                watchId,
                                currentFieldName
                        );
                    }
                }
                token = parser.nextToken();
            }
        }

        if (triggeredTime == null || scheduledTime == null) {
            throw new ElasticsearchParseException(
                    "could not parse trigger event for [{}] for watch [{}]. missing timestamp",
                    context,
                    watchId
            );
        }

        return new ScheduleTriggerEvent(triggeredTime, scheduledTime);
    }

    private static ZonedDateTime parseDateTime(String fieldName, XContentParser parser, Clock clock) throws IOException {
        try {
            return WatcherDateTimeUtils.parseDateMath(fieldName, parser, ZoneOffset.UTC, clock);
        } catch (ElasticsearchParseException pe) {
            throw new ElasticsearchParseException(
                    "could not parse trigger event. failed to parse date field [{}]",
                    pe,
                    fieldName
            );
        }
    }

    interface Field extends TriggerEvent.Field {
        ParseField SCHEDULED_TIME = new ParseField("scheduled_time");
    }
}

