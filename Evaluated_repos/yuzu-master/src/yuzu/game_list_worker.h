// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <memory>
#include <string>

#include <QList>
#include <QObject>
#include <QRunnable>
#include <QString>

#include "yuzu/compatibility_list.h"

namespace Core {
class System;
}

class QStandardItem;

namespace FileSys {
class NCA;
class VfsFilesystem;
} // namespace FileSys

/**
 * Asynchronous worker object for populating the game list.
 * Communicates with other threads through Qt's signal/slot system.
 */
class GameListWorker : public QObject, public QRunnable {
    Q_OBJECT

public:
    explicit GameListWorker(std::shared_ptr<FileSys::VfsFilesystem> vfs_,
                            FileSys::ManualContentProvider* provider_,
                            QVector<UISettings::GameDir>& game_dirs_,
                            const CompatibilityList& compatibility_list_, Core::System& system_);
    ~GameListWorker() override;

    /// Starts the processing of directory tree information.
    void run() override;

    /// Tells the worker that it should no longer continue processing. Thread-safe.
    void Cancel();

signals:
    /**
     * The `EntryReady` signal is emitted once an entry has been prepared and is ready
     * to be added to the game list.
     * @param entry_items a list with `QStandardItem`s that make up the columns of the new
     * entry.
     */
    void DirEntryReady(GameListDir* entry_items);
    void EntryReady(QList<QStandardItem*> entry_items, GameListDir* parent_dir);

    /**
     * After the worker has traversed the game directory looking for entries, this signal is
     * emitted with a list of folders that should be watched for changes as well.
     */
    void Finished(QStringList watch_list);

private:
    void AddTitlesToGameList(GameListDir* parent_dir);

    enum class ScanTarget {
        FillManualContentProvider,
        PopulateGameList,
    };

    void ScanFileSystem(ScanTarget target, const std::string& dir_path, bool deep_scan,
                        GameListDir* parent_dir);

    std::shared_ptr<FileSys::VfsFilesystem> vfs;
    FileSys::ManualContentProvider* provider;
    QVector<UISettings::GameDir>& game_dirs;
    const CompatibilityList& compatibility_list;

    QStringList watch_list;
    std::atomic_bool stop_processing;

    Core::System& system;
};
