/*
 * This class provides a tool for managing an Elasticsearch service on Windows.
 */

package org.elasticsearch.windows.service;

import org.elasticsearch.cli.CliToolProvider;
import org.elasticsearch.cli.Command;

public class WindowsServiceCliProvider implements CliToolProvider {

    // Returns the name of the tool
    @Override
    public String name() {
        return "windows-service";
    }

    // Creates a new Command object for the tool
    @Override
    public Command create() {
        return new WindowsServiceCli();
    }
    
}

