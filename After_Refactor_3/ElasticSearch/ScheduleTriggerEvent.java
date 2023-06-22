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
    private final ImmutableMap<String, Object> data;

    public ScheduleTriggerEvent(ZonedDateTime triggeredTime, ZonedDateTime scheduledTime) {
        this(null, triggeredTime, scheduledTime);
    }

    public ScheduleTriggerEvent(String jobName, ZonedDateTime triggeredTime, ZonedDateTime scheduledTime) {
        super(jobName, triggeredTime);
        this.scheduledTime = scheduledTime;
        // use immutable map to store data
        this.data = ImmutableMap.of(
            Field.SCHEDULED_TIME.getPreferredName(),
            ZonedDateTime.ofInstant(scheduledTime.toInstant(), ZoneOffset.UTC)
        );
    }

    public ZonedDateTime getScheduledTime() {
        return scheduledTime;
    }

    @Override
    public String type() {
        return ScheduleTrigger.TYPE;
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        WatcherDateTimeUtils.writeDate(Field.TRIGGERED_TIME.getPreferredName(), builder, getTriggeredTime());
        WatcherDateTimeUtils.writeDate(Field.SCHEDULED_TIME.getPreferredName(), builder, scheduledTime);
        return builder.endObject();
    }

    @Override
    public void recordDataXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject(ScheduleTrigger.TYPE);
        WatcherDateTimeUtils.writeDate(Field.SCHEDULED_TIME.getPreferredName(), builder, scheduledTime);
        builder.endObject();
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
                triggeredTime = parseDate(currentFieldName, parser, clock);
            } else if (Field.SCHEDULED_TIME.match(currentFieldName, parser.getDeprecationHandler())) {
                scheduledTime = parseDate(currentFieldName, parser, clock);
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

    // extract reusable code into a helper method
    private static ZonedDateTime parseDate(String fieldName, XContentParser parser, Clock clock) throws IOException {
        try {
            return WatcherDateTimeUtils.parseDateMath(fieldName, parser, ZoneOffset.UTC, clock);
        } catch (ElasticsearchParseException e) {
            throw new ElasticsearchParseException(
                "could not parse [{}] trigger event. failed to parse date field [{}]",
                e,
                ScheduleTriggerEngine.TYPE,
                fieldName
            );
        }
    }

    interface Field extends TriggerEvent.Field {
        ParseField SCHEDULED_TIME = new ParseField("scheduled_time");
    }
}