import base64
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from azure.cosmos.exceptions import CosmosHttpResponseError
from azure.cosmos.aio import CosmosClient as AsyncCosmosClient
from langgraph.checkpoint import BaseCheckpointSaver
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata, CheckpointTuple
from langchain_core.runnables import RunnableConfig
import json
from langgraph.checkpoint import BaseCheckpointSaver
from langgraph.serde.jsonplus import JsonPlusSerializer
from typing import (
    Any,
    AsyncGenerator,
    Generator,
    Optional
)
#from langchain import CheckpointManager
from dotenv import load_dotenv
load_dotenv("credentials.env")

class JsonAndBinarySerializer(JsonPlusSerializer):

    def _default(self, obj):
        if isinstance(obj, bytes):
            return {"type": "bytes", "data": base64.b64encode(obj).decode("utf-8")}
        if isinstance(obj, HumanMessage) or isinstance(obj, AIMessage) or isinstance(obj, ToolMessage):
            return {"type": type(obj).__name__, "data": obj.__dict__}
        if isinstance(obj, dict):
            return {k: self._default(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._default(v) for v in obj]
        return super()._default(obj)

    def dumps(self, obj):
        def serialize(obj):
            if isinstance(obj, (bytes, bytearray)):
                return {"type": "binary", "value": obj.hex()}
            elif isinstance(obj, HumanMessage):
                return obj.to_dict()
            elif isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [serialize(i) for i in obj]
            return obj

        return json.dumps(serialize(obj), default=self._default)

    def loads(self, s):
        def deserialize(obj):
            if isinstance(obj, dict):
                if "type" in obj and obj["type"] == "binary":
                    return bytes.fromhex(obj["value"])
                elif "_type" in obj and obj["_type"] == "HumanMessage":
                    return HumanMessage.from_dict(obj)
                return {k: deserialize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [deserialize(i) for i in obj]
            return obj

        return deserialize(json.loads(s))

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (HumanMessage, AIMessage, ToolMessage)):
            return obj.__dict__
        return super().default(obj)

# Initialize memory to persist state between graph runs
class CosmosSaver(BaseCheckpointSaver):
    def __init__(
        self,
        url: str,
        key: str,
        database_name: str,
        container_name: str,
    ):
        super().__init__(serde=JsonPlusSerializer())
        self.url = "https://localhost:8081/"
        self.key = "C2y6yDjf5/R+ob0N8A7Cgv30VRDJIWEHLM+4QDU5DE2nQ9nDuVTqobD4b8mGGyPMbIZnqyMsEcaGQy67XIw/Jw=="
        self.database_name = "visited"
        self.container_name = "emails"
        self.client = CosmosClient(self.url, self.key)
        self.async_client = AsyncCosmosClient(self.url, self.key)
        self._create_database_and_container()

    def _create_database_and_container(self):
        database = self.client.create_database_if_not_exists(self.database_name)
        database.create_container_if_not_exists(
            id=self.container_name,
            partition_key=PartitionKey(path="/emailid"),
            offer_throughput=400
        )

    async def _create_database_and_container_async(self):
        database = await self.async_client.create_database_if_not_exists(self.database_name)
        await database.create_container_if_not_exists(
            id=self.container_name,
            partition_key=PartitionKey(path="/emailid"),
            offer_throughput=400
        )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
    ) -> RunnableConfig:
        thread_id = config["configurable"]["thread_id"]
        thread_ts = checkpoint["ts"]
        parent_ts = config["configurable"].get("thread_ts")

        container = self.client.get_database_client(self.database_name).get_container_client(self.container_name)

        item = {
            "id": f"{thread_id}-{thread_ts}",
            "thread_id": thread_id,
            "thread_ts": thread_ts,
            "parent_ts": parent_ts,
            "checkpoint": json.dumps(checkpoint, cls = CustomJSONEncoder) if checkpoint else checkpoint,
            "metadata": json.dumps(metadata, cls = CustomJSONEncoder) if metadata else metadata
        }

        try:
            container.upsert_item(item)
        except CosmosHttpResponseError as e:
            print(f"Error upserting item: {e.message}")
            print(f"Serialized item: {item}")
            raise

        return {
            "configurable": {
                "thread_id": thread_id,
                "thread_ts": thread_ts,
            }
        }

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
    ) -> RunnableConfig:
        thread_id = config["configurable"]["thread_id"]
        thread_ts = checkpoint["ts"]
        parent_ts = config["configurable"].get("thread_ts")

        container = self.async_client.get_database_client(self.database_name).get_container_client(self.container_name)

        item = {
            "id": f"{thread_id}-{thread_ts}",
            "thread_id": thread_id,
            "thread_ts": thread_ts,
            "parent_ts": parent_ts,
            "checkpoint": json.dumps(checkpoint, cls = CustomJSONEncoder) if checkpoint else checkpoint,
            "metadata": json.dumps(metadata, cls = CustomJSONEncoder) if metadata else metadata
        }

        await container.upsert_item(item)

        return {
            "configurable": {
                "thread_id": thread_id,
                "thread_ts": thread_ts,
            },
        }

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Generator[CheckpointTuple, None, None]:
        thread_id = config["configurable"]["thread_id"]
        query = f"SELECT * FROM c WHERE c.thread_id = @thread_id"
        parameters = [{"name": "@thread_id", "value": thread_id}]

        if before:
            query += " AND c.thread_ts < @before_thread_ts"
            parameters.append({"name": "@before_thread_ts", "value": before["configurable"]["thread_ts"]})

        query += " ORDER BY c.thread_ts DESC"

        if limit:
            query += f" OFFSET 0 LIMIT {limit}"

        container = self.client.get_database_client(self.database_name).get_container_client(self.container_name)

        items = container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        )

        for item in items:
            yield CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": item["thread_id"],
                        "thread_ts": item["thread_ts"],
                    }
                },
                checkpoint=self.serde.loads(item["checkpoint"]),
                metadata=self.serde.loads(item["metadata"]),
                parent_config={
                    "configurable": {
                        "thread_id": item["thread_id"],
                        "thread_ts": item["parent_ts"],
                    }
                } if item.get("parent_ts") else None,
            )

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncGenerator[CheckpointTuple, None]:
        thread_id = config["configurable"]["thread_id"]
        query = f"SELECT * FROM c WHERE c.thread_id = @thread_id"
        parameters = [{"name": "@thread_id", "value": thread_id}]

        if before:
            query += " AND c.thread_ts < @before_thread_ts"
            parameters.append({"name": "@before_thread_ts", "value": before["configurable"]["thread_ts"]})

        query += " ORDER BY c.thread_ts DESC"

        if limit:
            query += f" OFFSET 0 LIMIT {limit}"

        container = self.async_client.get_database_client(self.database_name).get_container_client(self.container_name)

        async for item in container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        ):
            yield CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": item["thread_id"],
                        "thread_ts": item["thread_ts"],
                    }
                },
                checkpoint=self.serde.loads(item["checkpoint"]),
                metadata=self.serde.loads(item["metadata"]),
                parent_config={
                    "configurable": {
                        "thread_id": item["thread_id"],
                        "thread_ts": item["parent_ts"],
                    }
                } if item.get("parent_ts") else None,
            )

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        thread_id = config["configurable"]["thread_id"]
        thread_ts = config["configurable"].get("thread_ts")

        container = self.client.get_database_client(self.database_name).get_container_client(self.container_name)

        try:
            if thread_ts:
                item = container.read_item(item=f"{thread_id}-{thread_ts}", partition_key=thread_id)
            else:
                query = f"SELECT TOP 1 * FROM c WHERE c.thread_id = @thread_id ORDER BY c.thread_ts DESC"
                parameters = [{"name": "@thread_id", "value": thread_id}]
                items = list(container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True))
                item = items[0] if items else None

            if item:
                return CheckpointTuple(
                    config=config,
                    checkpoint=self.serde.loads(item["checkpoint"]),
                    metadata=self.serde.loads(item["metadata"]),
                    parent_config={
                        "configurable": {
                            "thread_id": item["thread_id"],
                            "thread_ts": item["parent_ts"],
                        }
                    } if item.get("parent_ts") else None,
                )
        except exceptions.CosmosResourceNotFoundError:
            return None

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        thread_id = config["configurable"]["thread_id"]
        thread_ts = config["configurable"].get("thread_ts")

        container = self.async_client.get_database_client(self.database_name).get_container_client(self.container_name)

        try:
            if thread_ts:
                item = await container.read_item(item=f"{thread_id}-{thread_ts}", partition_key=thread_id)
            else:
                query = f"SELECT TOP 1 * FROM c WHERE c.thread_id = @thread_id ORDER BY c.thread_ts DESC"
                parameters = [{"name": "@thread_id", "value": thread_id}]
                items = [item async for item in container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True)]
                item = items[0] if items else None

            if item:
                return CheckpointTuple(
                    config=config,
                    checkpoint=self.serde.loads(item["checkpoint"]),
                    metadata=self.serde.loads(item["metadata"]),
                    parent_config={
                        "configurable": {
                            "thread_id": item["thread_id"],
                            "thread_ts": item["parent_ts"],
                        }
                    } if item.get("parent_ts") else None,
                )
        except exceptions.CosmosResourceNotFoundError:
            return None
