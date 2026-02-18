"""Pydantic models for Catalyst API requests and responses."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ProjectMetadata(BaseModel):
    name: str | None = Field(default=None)


class ProjectSpec(BaseModel):
    agent_infrastructure_enabled: bool | None = Field(
        default=None, alias="agentInfrastructureEnabled"
    )

    model_config = {"populate_by_name": True}


class ProjectEndpointDetails(BaseModel):
    url: str | None = Field(default=None)
    port: int | None = Field(default=None)


class ProjectEndpoints(BaseModel):
    http: ProjectEndpointDetails | None = Field(default=None)
    grpc: ProjectEndpointDetails | None = Field(default=None)


class ProjectStatus(BaseModel):
    endpoints: ProjectEndpoints | None = Field(default=None)
    status: str | None = Field(default=None)


class Project(BaseModel):
    metadata: ProjectMetadata | None = Field(default=None)
    spec: ProjectSpec | None = Field(default=None)
    status: ProjectStatus | None = Field(default=None)


class CreateProjectRequest(BaseModel):
    name: str
    agent_infrastructure_enabled: bool = Field(
        default=False, alias="agentInfrastructureEnabled"
    )

    model_config = {"populate_by_name": True}


class AppIDMetadata(BaseModel):
    name: str | None = Field(default=None)


class AppIDSpec(BaseModel):
    app_port: int | None = Field(default=None, alias="appPort")

    model_config = {"populate_by_name": True}


class AppIDStatus(BaseModel):
    api_token: str | None = Field(default=None, alias="apiToken")
    status: str | None = Field(default=None)

    model_config = {"populate_by_name": True}


class AppID(BaseModel):
    metadata: AppIDMetadata | None = Field(default=None)
    spec: AppIDSpec | None = Field(default=None)
    status: AppIDStatus | None = Field(default=None)


class CreateAppIDRequest(BaseModel):
    name: str
    app_port: int | None = Field(default=None, alias="appPort")

    model_config = {"populate_by_name": True}
