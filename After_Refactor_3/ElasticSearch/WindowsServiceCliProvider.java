/**
 * A provider for the Elasticsearch Windows service management CLI tool.
 */
package org.elasticsearch.windows.service;

import org.elasticsearch.cli.CliToolProvider;
import org.elasticsearch.cli.Command;

/**
 * Provides a tool for managing an Elasticsearch service on Windows.
 */
public class WindowsServiceCliProvider implements CliToolProvider {
    
    /**
     * Returns the name of the CLI tool.
     */
    @Override
    public String name() {
        return "windows-service";
    }
    
    /**
     * Creates a new command for the CLI tool.
     */
    @Override
    public Command create() {
        return new WindowsServiceCliCommand();
    }
}
