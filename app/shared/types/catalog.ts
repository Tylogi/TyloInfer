import { type Dispatch, type FormEvent, type RefObject, type SetStateAction } from 'react';
import type { ModelArtifact, ModelDeletionPlan, ModelDirectoryList, RuntimeInstance, RuntimeStatus } from './api';
import { type LoadContextDefaults } from './load-context';

export interface ModelCatalog {
    connectionRevision: number;
    runtime: RuntimeStatus | null;
    artifacts: ModelArtifact[];
    busy: boolean;
    ready: boolean;
    instances: RuntimeInstance[];
    availableModelNames: string[];
    modelFilter: string;
    setModelFilter: Dispatch<SetStateAction<string>>;
    filteredInstances: RuntimeInstance[];
    filteredArtifacts: ModelArtifact[];
    loadPinned: boolean;
    loadContext: LoadContextDefaults;
    loadContextForModel: (name: string) => LoadContextDefaults;
    setLoadContext: (value: LoadContextDefaults) => void;
    loadDeviceIds: string[];
    setLoadDeviceIds: (deviceIds: string[]) => void;
    setLoadPinned: (pinned: boolean) => void;
    loadIdleTtl: number | null;
    setLoadIdleTtl: (idleTtl: number | null) => void;
    modelBrowser: ModelDirectoryList | null;
    modelBrowserOpen: boolean;
    modelFolderPath: string;
    modelBrowserError: string;
    modelBrowserLoading: boolean;
    modelFilesMode: boolean;
    setModelBrowserOpen: (open: boolean) => void;
    modelDirectoryPath: string;
    setModelDirectoryPath: Dispatch<SetStateAction<string>>;
    modelBrowserTriggerRef: RefObject<HTMLElement | null>;
    openStudioPage: (_view: string, page: 'models') => void | Promise<void>;
    chooseModelDirectory: () => Promise<void>;
    jumpToModelDirectory: (event: FormEvent<HTMLFormElement>) => Promise<void>;
    openModelDirectory: (directoryId?: string | null, path?: string | null) => Promise<void>;
    openModelFiles: (modelId: string) => Promise<void>;
    openCurrentDirectory: () => Promise<void>;
    registerCurrentModelDirectory: () => Promise<void>;
    unloadInstance: (id: string) => Promise<void>;
    loadArtifact: (name: string, companions?: string[], draft?: {
        id: string;
        instanceId?: string;
    }) => Promise<void>;
    modelDeletion: ModelDeletionPlan | null;
    deletionTriggerRef: RefObject<HTMLElement | null>;
    prepareModelDeletion: (modelId: string) => Promise<void>;
    cancelModelDeletion: () => void;
    confirmModelDeletion: () => Promise<void>;
    settingsModel: string | null;
    setSettingsModel: Dispatch<SetStateAction<string | null>>;
    openModelSettings: (name: string) => void;
    modelSettingsTriggerRef: RefObject<HTMLElement | null>;

}
