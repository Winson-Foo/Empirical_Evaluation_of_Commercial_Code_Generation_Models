/*
 * This file is part of the Elasticsearch project.
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the Server Side Public License, v 1; you may not use this file except
 * in compliance with, at your election, the Elastic License 2.0 or the Server
 * Side Public License, v 1.
 */

package org.elasticsearch.windows.service;

import org.elasticsearch.cli.CliToolProvider;
import org.elasticsearch.cli.Command;

/**
 * A provider for a command-line interface tool that manages an Elasticsearch service on Windows.
 */
public class ElasticsearchWindowsServiceCliProvider implements CliToolProvider {

    /**
     * Returns the name of the command-line interface tool provided by this provider.
     * @return The name of the command-line interface tool.
     */
    @Override
    public String name() {
        return "elasticsearch-windows-service";
    }

    /**
     * Creates and returns an instance of the command-line interface tool provided by this provider.
     * @return An instance of the command-line interface tool.
     */
    @Override
    public Command create() {
        return new ElasticsearchWindowsServiceCli();
    }
}

