"use client";

import { Suspense, useRef, useMemo, useEffect, useState, useCallback, memo, type ComponentRef, type MutableRefObject } from "react";
import * as THREE from "three";
import { Canvas, invalidate, useFrame, type ThreeEvent } from "@react-three/fiber";
import {
  CameraControls,
  ContactShadows,
  Environment,
  MeshReflectorMaterial,
  useGLTF,
  useTexture,
} from "@react-three/drei";
import { DecalGeometry } from "three-stdlib";
import { Model as RenaultMasterPanelVan } from "@/Renault_master_panel_van";
import { RenaultMasterCollider } from "@/Renault_master_collider";

type CameraControlsRef = ComponentRef<typeof CameraControls>;

const CAMERA_TUNING = {
  sensitivityRefDistance: 12,
  minSensitivity: 0.5,
  maxSensitivity: 1,
  truckDistanceFactor: 12,
  minTruckSpeed: 1,
  maxTruckSpeed: 2,
  dollyDistanceFactor: 12,
  minDollySpeed: 1,
  maxDollySpeed: 0.5,
} as const;

const CAMERA_BOUNDS = {
  minX: -2,
  maxX: 2,
  minY: -2,
  maxY: 2,
  minZ: -4,
  maxZ: 4,
} as const;

const SHOW_CAMERA_BOUNDS_GIZMO = true;
const ENABLE_UV_HOVER = false;
const PIXEL_MAPPING_MODE: "uv" | "surface" = "surface";
const USE_SIMPLIFIED_COLLIDER_FOR_PICKING = false;
const COLLIDER_MODEL_PATH = "/renault_master_collider.glb";
const GRID_U = 50;
const GRID_V = 50;
const OVERLAY_CELL_SIZE = 16;
const OVERLAY_WIDTH = GRID_U * OVERLAY_CELL_SIZE;
const OVERLAY_HEIGHT = GRID_V * OVERLAY_CELL_SIZE;
const SURFACE_PIXEL_WORLD_SIZE = .3;
const SURFACE_PIXEL_OFFSET = 0.015;
const PIXELS_PER_FACE_EDGE = 12;
const ENABLE_SURFACE_DECALS = true;
const USE_SHADER_PAINTING = true; // Usa decals enquanto ajustamos o mapeamento da grid na malha
const FORCE_SURFACE_DECAL_FALLBACK = false; // Mostra decals só como fallback visual opcional
const MAX_ACTIVE_SURFACE_DECALS = 700;
const SHOW_SURFACE_PIXEL_DEBUG_DOT = false;
const SHOW_DEBUG_GRID = true;
const USE_COLLIDER_AS_VISUAL_TEST = true;
const MIN_GRID_CELLS_PER_AXIS = 4; // Garante pelo menos N células em cada eixo projetado (evita linhas faltantes)
const MIN_SURFACE_ANGLE_DEGREES = 30; // Ângulo mínimo entre normal e plano do grid (evita células muito esticadas)

useGLTF.preload(COLLIDER_MODEL_PATH);

type HoveredCell = {
  meshUuid: string;
  meshName: string;
  faceIndex: number;
  gridU: number;
  gridV: number;
  key: string;
};

type MeshInfo = {
  uuid: string;
  name: string;
  partName: string;
};

type SurfacePixel = {
  key: string;
  partName: string;
  meshUuid: string;
  meshName: string;
  plane: "xy" | "xz" | "yz";
  planeCoord1: number;
  planeCoord2: number;
  planeCoord3: number;
  sharedPlane?: "xy" | "xz" | "yz";
  sharedPlaneCoord1?: number;
  sharedPlaneCoord2?: number;
  faceIndex: number;
  gridU: number;
  gridV: number;
  uv?: [number, number];
  position: [number, number, number];
  quaternion: [number, number, number, number];
};

type SurfaceClickDebug = {
  meshName: string;
  partName: string;
  plane: "xy" | "xz" | "yz";
  u: number;
  v: number;
  depth: number;
  key: string;
  action: "add" | "remove";
};

type SurfaceProjection = {
  plane: "xy" | "xz" | "yz";
  axisA: "x" | "y" | "z";
  axisB: "x" | "y" | "z";
  normalAxis: "x" | "y" | "z";
  originA: number;
  originB: number;
};

type SurfaceGridSnap = {
  snappedPoint: THREE.Vector3;
  gridU: number;
  gridV: number;
};

type SurfaceDecalMeshProps = {
  pixel: SurfacePixel;
  targetMesh: THREE.Mesh;
};

function forEachStandardMaterial(
  material: THREE.Material | THREE.Material[],
  callback: (mat: THREE.MeshStandardMaterial) => void,
) {
  if (Array.isArray(material)) {
    material.forEach((mat) => {
      if (mat instanceof THREE.MeshStandardMaterial) callback(mat);
    });
    return;
  }

  if (material instanceof THREE.MeshStandardMaterial) callback(material);
}

function cloneMeshMaterialsOnce(mesh: THREE.Mesh) {
  if (mesh.userData.pixelMaterialCloned) return;

  if (Array.isArray(mesh.material)) {
    mesh.material = mesh.material.map((mat) => mat.clone());
  } else {
    mesh.material = mesh.material.clone();
  }

  mesh.userData.pixelMaterialCloned = true;
}

function clampCellIndex(value: number, max: number) {
  return Math.min(max - 1, Math.max(0, value));
}

function uvToCell(uv: THREE.Vector2) {
  const u = clampCellIndex(Math.floor(uv.x * GRID_U), GRID_U);
  const v = clampCellIndex(Math.floor(uv.y * GRID_V), GRID_V);
  return { u, v };
}

function cellKey(partName: string, u: number, v: number) {
  return `${partName}:${u}:${v}`;
}

function getAxisValue(v: THREE.Vector3, axis: "x" | "y" | "z") {
  if (axis === "x") return v.x;
  if (axis === "y") return v.y;
  return v.z;
}

function setAxisValue(v: THREE.Vector3, axis: "x" | "y" | "z", value: number) {
  if (axis === "x") v.x = value;
  else if (axis === "y") v.y = value;
  else v.z = value;
}

function getDominantNormalAxis(normal: THREE.Vector3): "x" | "y" | "z" {
  const absX = Math.abs(normal.x);
  const absY = Math.abs(normal.y);
  const absZ = Math.abs(normal.z);
  if (absX >= absY && absX >= absZ) return "x";
  if (absY >= absZ) return "y";
  return "z";
}

function projectionFromNormalAxis(normalAxis: "x" | "y" | "z", boundingBox: THREE.Box3): SurfaceProjection {
  if (normalAxis === "x") {
    return {
      plane: "yz",
      axisA: "y",
      axisB: "z",
      normalAxis,
      originA: boundingBox.min.y,
      originB: boundingBox.min.z,
    };
  }

  if (normalAxis === "y") {
    return {
      plane: "xz",
      axisA: "x",
      axisB: "z",
      normalAxis,
      originA: boundingBox.min.x,
      originB: boundingBox.min.z,
    };
  }

  return {
    plane: "xy",
    axisA: "x",
    axisB: "y",
    normalAxis,
    originA: boundingBox.min.x,
    originB: boundingBox.min.y,
  };
}

function getMeshSurfaceProjection(mesh: THREE.Mesh): SurfaceProjection {
  const cachedProjection = mesh.userData.surfaceProjection as SurfaceProjection | undefined;
  if (cachedProjection) return cachedProjection;

  const geometry = mesh.geometry;
  if (!geometry.boundingBox) geometry.computeBoundingBox();
  const boundingBox = geometry.boundingBox ?? new THREE.Box3(
    new THREE.Vector3(-0.5, -0.5, -0.5),
    new THREE.Vector3(0.5, 0.5, 0.5),
  );

  // Determine dominant normal axis by accumulating absolute normal components
  // across all vertices. This correctly handles curved/angled surfaces where the
  // bounding-box "thin axis" approach would pick the wrong projection plane
  // (e.g. a door that is curved outward = bounding box is NOT thin in X, but
  // face normals still mostly point in X).
  let normalAxis: "x" | "y" | "z" = "z";
  const normalAttr = geometry.getAttribute("normal") as THREE.BufferAttribute | undefined;
  if (normalAttr && normalAttr.count > 0) {
    let nx = 0, ny = 0, nz = 0;
    // Sample up to 512 evenly-spaced vertices for performance.
    const step = Math.max(1, Math.floor(normalAttr.count / 512));
    for (let i = 0; i < normalAttr.count; i += step) {
      nx += Math.abs(normalAttr.getX(i));
      ny += Math.abs(normalAttr.getY(i));
      nz += Math.abs(normalAttr.getZ(i));
    }
    if (nx >= ny && nx >= nz) normalAxis = "x";
    else if (ny >= nz) normalAxis = "y";
    else normalAxis = "z";
  } else {
    // Fallback: bounding-box thin axis (original behaviour).
    const size = new THREE.Vector3();
    boundingBox.getSize(size);
    if (size.x <= size.y && size.x <= size.z) normalAxis = "x";
    else if (size.y <= size.z) normalAxis = "y";
    else normalAxis = "z";
  }

  const projection = projectionFromNormalAxis(normalAxis, boundingBox);

  mesh.userData.surfaceProjection = projection;
  return projection;
}

function getMeshLocalCellSize(
  mesh: THREE.Mesh,
  projection: SurfaceProjection,
  worldCellSize: number,
) {
  const worldScale = mesh.getWorldScale(new THREE.Vector3());
  const axisAScale = Math.abs(getAxisValue(worldScale, projection.axisA));
  const axisBScale = Math.abs(getAxisValue(worldScale, projection.axisB));
  const projectedScale = Math.max((axisAScale + axisBScale) * 0.5, 1e-6);

  const rawCellSize = worldCellSize / projectedScale;

  // Garante que pelo menos MIN_GRID_CELLS_PER_AXIS células caibam em cada eixo projetado.
  // Sem isso, peças estreitas (portas, capôs) ficam sem linhas em uma direção.
  const geometry = mesh.geometry;
  if (!geometry.boundingBox) geometry.computeBoundingBox();
  const bboxSize = new THREE.Vector3();
  geometry.boundingBox!.getSize(bboxSize);
  const extentA = Math.abs(getAxisValue(bboxSize, projection.axisA));
  const extentB = Math.abs(getAxisValue(bboxSize, projection.axisB));
  const maxFromA = extentA > 0 ? extentA / MIN_GRID_CELLS_PER_AXIS : rawCellSize;
  const maxFromB = extentB > 0 ? extentB / MIN_GRID_CELLS_PER_AXIS : rawCellSize;

  return Math.min(rawCellSize, maxFromA, maxFromB);
}

function getCachedMeshLocalCellSize(
  mesh: THREE.Mesh,
  projection: SurfaceProjection,
  worldCellSize: number,
) {
  const cacheByPlane = (mesh.userData.surfaceLocalCellSizeByPlane as Partial<Record<"xy" | "xz" | "yz", number>> | undefined) ?? {};
  const cachedCellSize = cacheByPlane[projection.plane];
  if (Number.isFinite(cachedCellSize) && (cachedCellSize as number) > 0) return cachedCellSize as number;

  const computedCellSize = getMeshLocalCellSize(mesh, projection, worldCellSize);
  cacheByPlane[projection.plane] = computedCellSize;
  mesh.userData.surfaceLocalCellSizeByPlane = cacheByPlane;
  return computedCellSize;
}

function snapPointToSurfaceGrid(
  point: THREE.Vector3,
  projection: SurfaceProjection,
  cellSize: number,
): SurfaceGridSnap {
  const coordA = getAxisValue(point, projection.axisA);
  const coordB = getAxisValue(point, projection.axisB);

  const gridU = Math.floor((coordA - projection.originA) / cellSize);
  const gridV = Math.floor((coordB - projection.originB) / cellSize);

  const snappedPoint = point.clone();
  setAxisValue(snappedPoint, projection.axisA, projection.originA + ((gridU + 0.5) * cellSize));
  setAxisValue(snappedPoint, projection.axisB, projection.originB + ((gridV + 0.5) * cellSize));

  return {
    snappedPoint,
    gridU,
    gridV,
  };
}

function isSurfaceAngleValid(normal: THREE.Vector3): boolean {
  // Verifica se o ângulo entre a normal e o plano dominante é adequado
  // Evita células muito esticadas em superfícies quase paralelas ao grid
  const absX = Math.abs(normal.x);
  const absY = Math.abs(normal.y);
  const absZ = Math.abs(normal.z);
  
  const maxComponent = Math.max(absX, absY, absZ);
  const minAngleRad = (MIN_SURFACE_ANGLE_DEGREES * Math.PI) / 180;
  const minDot = Math.sin(minAngleRad); // sin porque queremos o ângulo com o plano, não com a normal do plano
  
  // Se a componente dominante é maior que o threshold, o ângulo é ok
  return maxComponent >= minDot;
}

function getMeshDisplayName(mesh: THREE.Mesh) {
  return mesh.name || mesh.parent?.name || "mesh";
}

function isMeshObject(obj: THREE.Object3D): obj is THREE.Mesh {
  return (obj as THREE.Mesh).isMesh === true;
}

const SurfaceDecalMesh = memo(function SurfaceDecalMesh({
  pixel,
  targetMesh,
}: SurfaceDecalMeshProps) {
  const geometry = useMemo(() => {
    const position = new THREE.Vector3(...pixel.position);
    const rotation = new THREE.Euler().setFromQuaternion(new THREE.Quaternion(...pixel.quaternion));
    const scale = new THREE.Vector3(
      SURFACE_PIXEL_WORLD_SIZE,
      SURFACE_PIXEL_WORLD_SIZE,
      SURFACE_PIXEL_WORLD_SIZE * 1.8,
    );

    return new DecalGeometry(targetMesh, position, rotation, scale);
  }, [targetMesh, pixel.key]);

  const material = useMemo(() => {
    // Material customizado com clipping para evitar vazamento entre células
    return new THREE.ShaderMaterial({
      uniforms: {
        color: { value: new THREE.Color("#ff2d55") },
        opacity: { value: 0.92 },
        cellCenter: { value: new THREE.Vector3(...pixel.position) },
        cellHalfSize: { value: SURFACE_PIXEL_WORLD_SIZE * 0.5 },
      },
      vertexShader: `
        varying vec3 vWorldPosition;
        void main() {
          vec4 worldPosition = modelMatrix * vec4(position, 1.0);
          vWorldPosition = worldPosition.xyz;
          gl_Position = projectionMatrix * viewMatrix * worldPosition;
        }
      `,
      fragmentShader: `
        uniform vec3 color;
        uniform float opacity;
        uniform vec3 cellCenter;
        uniform float cellHalfSize;
        varying vec3 vWorldPosition;
        
        void main() {
          // Clipping: descarta fragmentos fora da célula do grid
          vec3 delta = abs(vWorldPosition - cellCenter);
          if (delta.x > cellHalfSize || delta.y > cellHalfSize || delta.z > cellHalfSize) {
            discard;
          }
          
          gl_FragColor = vec4(color, opacity);
        }
      `,
      transparent: true,
      depthWrite: false,
      depthTest: true,
      side: THREE.DoubleSide,
      toneMapped: false,
    });
  }, [pixel.position, pixel.key]);

  useEffect(() => {
    return () => {
      geometry.dispose();
      material.dispose();
    };
  }, [geometry, material]);

  return (
    <mesh geometry={geometry} material={material} frustumCulled renderOrder={9998} />
  );
}, (prev, next) => prev.pixel === next.pixel && prev.targetMesh === next.targetMesh);

function ColliderProxyModel({
  modelPath,
  onPointerMove,
  onPointerOut,
  onPointerDown,
  onMeshRegistry,
}: {
  modelPath: string;
  onPointerMove?: (e: ThreeEvent<PointerEvent>) => void;
  onPointerOut?: () => void;
  onPointerDown: (e: ThreeEvent<PointerEvent>) => void;
  onMeshRegistry?: (meshes: MeshInfo[]) => void;
}) {
  const { scene } = useGLTF(modelPath);
  const colliderScene = useMemo(() => scene.clone(true), [scene]);

  useEffect(() => {
    colliderScene.traverse((obj) => {
      if (!isMeshObject(obj)) return;

      const toColliderMaterial = (mat: THREE.Material) => {
        const clone = mat.clone();
        clone.transparent = true;
        clone.opacity = 0;
        clone.depthWrite = false;
        clone.colorWrite = false;
        clone.side = THREE.DoubleSide;
        clone.needsUpdate = true;
        return clone;
      };

      if (Array.isArray(obj.material)) {
        obj.material = obj.material.map(toColliderMaterial);
      } else {
        obj.material = toColliderMaterial(obj.material);
      }
    });
  }, [colliderScene]);

  useEffect(() => {
    if (!onMeshRegistry) return;

    const meshes: MeshInfo[] = [];
    colliderScene.traverse((obj) => {
      if (isMeshObject(obj)) {
        const name = obj.name || obj.parent?.name || "mesh";
        meshes.push({ uuid: obj.uuid, name, partName: obj.uuid });
      }
    });
    onMeshRegistry(meshes);
  }, [colliderScene, onMeshRegistry]);

  return (
    <group
      onPointerMove={onPointerMove}
      onPointerOut={onPointerOut}
      onPointerDown={onPointerDown}
    >
      <primitive
        object={colliderScene}
        scale={0.02}
        rotation={[-Math.PI / 2, 0, 0]}
        position={[0, -2.5, 0]}
      />
    </group>
  );
}

function surfaceQuaternionFromNormal(normal: THREE.Vector3) {
  const absX = Math.abs(normal.x);
  const absY = Math.abs(normal.y);
  const absZ = Math.abs(normal.z);

  // Quantiza para 6 direções principais para orientação consistente
  let dominantNormal: THREE.Vector3;
  
  if (absX > absY && absX > absZ) {
    // Plano YZ (normal em X)
    dominantNormal = new THREE.Vector3(Math.sign(normal.x) || 1, 0, 0);
  } else if (absY > absX && absY > absZ) {
    // Plano XZ (normal em Y)
    dominantNormal = new THREE.Vector3(0, Math.sign(normal.y) || 1, 0);
  } else {
    // Plano XY (normal em Z)
    dominantNormal = new THREE.Vector3(0, 0, Math.sign(normal.z) || 1);
  }

  const worldUp = new THREE.Vector3(0, 1, 0);
  const worldForward = new THREE.Vector3(0, 0, 1);

  let up = worldUp.clone().sub(dominantNormal.clone().multiplyScalar(worldUp.dot(dominantNormal)));
  if (up.lengthSq() < 1e-6) {
    up = worldForward.clone().sub(dominantNormal.clone().multiplyScalar(worldForward.dot(dominantNormal)));
  }
  up.normalize();

  const right = new THREE.Vector3().crossVectors(up, dominantNormal).normalize();
  const correctedUp = new THREE.Vector3().crossVectors(dominantNormal, right).normalize();

  const basis = new THREE.Matrix4().makeBasis(right, correctedUp, dominantNormal);
  return new THREE.Quaternion().setFromRotationMatrix(basis);
}

function CameraRig({
  controlsRef,
}: {
  controlsRef: MutableRefObject<CameraControlsRef | null>;
}) {
  const hasInitializedLimits = useRef(false);
  const frameCount = useRef(0);

  useFrame(() => {
    const controls = controlsRef.current;
    if (!controls) return;

    if (!hasInitializedLimits.current) {
      controls.minPolarAngle = Math.PI / 4;
      controls.maxPolarAngle = Math.PI / 1.5;
      controls.setBoundary(
        new THREE.Box3(
          new THREE.Vector3(CAMERA_BOUNDS.minX, CAMERA_BOUNDS.minY, CAMERA_BOUNDS.minZ),
          new THREE.Vector3(CAMERA_BOUNDS.maxX, CAMERA_BOUNDS.maxY, CAMERA_BOUNDS.maxZ),
        ),
      );
      controls.boundaryEnclosesCamera = false;
      hasInitializedLimits.current = true;
    }

    // Otimização: atualiza controles apenas a cada 3 frames
    frameCount.current++;
    if (frameCount.current % 3 !== 0) return;

    const dist = controls.distance;
    const rotationSpeed = THREE.MathUtils.clamp(
      dist / CAMERA_TUNING.sensitivityRefDistance,
      CAMERA_TUNING.minSensitivity,
      CAMERA_TUNING.maxSensitivity,
    );

    controls.azimuthRotateSpeed = rotationSpeed;
    controls.polarRotateSpeed = rotationSpeed;
    controls.truckSpeed = THREE.MathUtils.clamp(
      dist / CAMERA_TUNING.truckDistanceFactor,
      CAMERA_TUNING.minTruckSpeed,
      CAMERA_TUNING.maxTruckSpeed,
    );
    controls.dollySpeed = THREE.MathUtils.clamp(
      dist / CAMERA_TUNING.dollyDistanceFactor,
      CAMERA_TUNING.minDollySpeed,
      CAMERA_TUNING.maxDollySpeed,
    );
  });

  return null;
}

function VanModel({
  onPointerMove,
  onPointerOut,
  onPointerDown,
  onMeshRegistry,
  onMeshesReady,
  useColliderProxy,
  colliderModelPath,
  paintedCells,
}: {
  onPointerMove?: (e: ThreeEvent<PointerEvent>) => void;
  onPointerOut?: () => void;
  onPointerDown: (e: ThreeEvent<PointerEvent>) => void;
  onMeshRegistry?: (meshes: MeshInfo[]) => void;
  onMeshesReady?: (meshes: THREE.Mesh[]) => void;
  useColliderProxy?: boolean;
  colliderModelPath?: string;
  paintedCells?: Map<string, SurfacePixel>;
}) {
  const visualGroupRef = useRef<THREE.Group>(null);
  const hasScannedMeshesRef = useRef(false);
  const frameCountRef = useRef(0);

  console.log("🚀 VanModel component mounted/rendered");

  // Use useFrame to detect meshes after GLTF loads into scene graph
  useFrame(() => {
    frameCountRef.current++;
    
    // Log every 60 frames to show it's running
    if (frameCountRef.current % 60 === 1) {
      const childCount = visualGroupRef.current?.children.length ?? 0;
      console.log(`⏱️ Frame ${frameCountRef.current}: useFrame is running, ref exists: ${!!visualGroupRef.current}, children: ${childCount}, already scanned: ${hasScannedMeshesRef.current}`);
    }
    
    if (hasScannedMeshesRef.current || !visualGroupRef.current) return;

    const meshObjects: THREE.Mesh[] = [];
    const meshes: MeshInfo[] = [];
    let objectCount = 0;
    
    console.log(`🔍 Starting traverse, group has ${visualGroupRef.current.children.length} direct children`);
    
    visualGroupRef.current.traverse((obj) => {
      objectCount++;
      if (frameCountRef.current < 120 || (objectCount <= 5)) {
        console.log(`  👁️ Object ${objectCount}: ${obj.type} "${obj.name || '(unnamed)'}"`);
      }
      if (isMeshObject(obj)) {
        const name = obj.name || obj.parent?.name || "mesh";
        console.log(`  ✅ FOUND MESH: "${name}" (uuid: ${obj.uuid})`);
        meshes.push({ uuid: obj.uuid, name, partName: obj.uuid });
        meshObjects.push(obj);

        if (obj.material && (SHOW_DEBUG_GRID || USE_SHADER_PAINTING)) {
          cloneMeshMaterialsOnce(obj);
        }
        
        // Aplica shader quando grid debug ou shader painting estiver habilitado
        if ((SHOW_DEBUG_GRID || USE_SHADER_PAINTING) && obj.material) {
          if (!obj.geometry.boundingBox) obj.geometry.computeBoundingBox();
          const bbox = obj.geometry.boundingBox!;
          const projectionYZ = projectionFromNormalAxis("x", bbox);
          const projectionXZ = projectionFromNormalAxis("y", bbox);
          const projectionXY = projectionFromNormalAxis("z", bbox);
          const gridSizeYZ = getCachedMeshLocalCellSize(obj, projectionYZ, SURFACE_PIXEL_WORLD_SIZE);
          const gridSizeXZ = getCachedMeshLocalCellSize(obj, projectionXZ, SURFACE_PIXEL_WORLD_SIZE);
          const gridSizeXY = getCachedMeshLocalCellSize(obj, projectionXY, SURFACE_PIXEL_WORLD_SIZE);
          const materials = Array.isArray(obj.material) ? obj.material : [obj.material];
          
          materials.forEach((mat, idx) => {
            console.log(`    🎛️ Material ${idx} type: ${mat.type}`);
            if (!mat.userData.hasGridOverlay) {
              console.log(`    ⚙️ Installing grid shader on material ${idx}`);
              // FORÇA recompilação removendo program cache
              mat.needsUpdate = true;
              const programCacheKey = `grid-overlay:${obj.uuid}:${idx}`;
              (mat as THREE.Material & { customProgramCacheKey?: () => string }).customProgramCacheKey = () => programCacheKey;
              if ((mat as any).program) {
                delete (mat as any).program;
              }
              
              mat.onBeforeCompile = (shader: any) => {
                console.log(`    ✅ onBeforeCompile fired for material ${idx}`);

                // Cria textura para armazenar células pintadas (mais eficiente que array de uniforms)
                const dataTexture = new THREE.DataTexture(
                  new Float32Array(700 * 4), // 700 células, 4 componentes RGBA
                  700,
                  1,
                  THREE.RGBAFormat,
                  THREE.FloatType
                );
                dataTexture.magFilter = THREE.NearestFilter;
                dataTexture.minFilter = THREE.NearestFilter;
                dataTexture.wrapS = THREE.ClampToEdgeWrapping;
                dataTexture.wrapT = THREE.ClampToEdgeWrapping;
                dataTexture.generateMipmaps = false;
                dataTexture.needsUpdate = true;
                
                shader.uniforms.gridSizeByPlane = { value: new THREE.Vector3(gridSizeYZ, gridSizeXZ, gridSizeXY) };
                shader.uniforms.gridColor = { value: new THREE.Color(0x00ff00) };
                shader.uniforms.gridOpacity = { value: SHOW_DEBUG_GRID ? 0.85 : 0.0 };
                shader.uniforms.paintColor = { value: new THREE.Color(0xff2d55) };
                shader.uniforms.gridOriginMin = { value: new THREE.Vector3(bbox.min.x, bbox.min.y, bbox.min.z) };
                shader.uniforms.useShaderPainting = { value: USE_SHADER_PAINTING };
                shader.uniforms.paintedCellsTexture = { value: dataTexture };
                shader.uniforms.numPaintedCells = { value: 0 };
                
                shader.vertexShader = shader.vertexShader.replace(
                  '#include <common>',
                  `#include <common>
                  varying vec3 vWorldPosition;
                  varying vec3 vLocalPosition;
                  varying vec3 vLocalNormal;`
                );
                
                shader.vertexShader = shader.vertexShader.replace(
                  '#include <begin_vertex>',
                  `#include <begin_vertex>
                  vWorldPosition = (modelMatrix * vec4(position, 1.0)).xyz;
                  vLocalPosition = position;
                  vLocalNormal = normal;`
                );
                
                shader.fragmentShader = shader.fragmentShader.replace(
                  '#include <common>',
                  `#include <common>
                  varying vec3 vWorldPosition;
                  varying vec3 vLocalPosition;
                  varying vec3 vLocalNormal;
                  uniform vec3 gridSizeByPlane;
                  uniform vec3 gridColor;
                  uniform float gridOpacity;
                  uniform vec3 paintColor;
                  uniform vec3 gridOriginMin;
                  uniform bool useShaderPainting;
                  uniform sampler2D paintedCellsTexture;
                  uniform int numPaintedCells;`
                );
                
                shader.fragmentShader = shader.fragmentShader.replace(
                  '#include <tonemapping_fragment>',
                  `#include <tonemapping_fragment>

                  // Seleciona o plano por fragmento usando a normal local dominante.
                  // Isso cobre melhor regiões com ângulos muito diferentes na mesma peça.
                  vec3 geometricNormal = normalize(cross(dFdx(vLocalPosition), dFdy(vLocalPosition)));
                  vec3 absN = abs(geometricNormal);
                  if (absN.x + absN.y + absN.z < 1e-5) {
                    absN = abs(vLocalNormal);
                  }
                  vec2 surfaceCoords;
                  vec2 effectiveOrigin;
                  float fragmentPlaneId;
                  float currentGridSize;
                  float normalSide;
                  if (absN.x >= absN.y && absN.x >= absN.z) {
                    surfaceCoords = vLocalPosition.yz;
                    effectiveOrigin = gridOriginMin.yz;
                    fragmentPlaneId = 0.0;
                    currentGridSize = max(gridSizeByPlane.x, 1e-6);
                    normalSide = geometricNormal.x >= 0.0 ? 1.0 : 0.0;
                  } else if (absN.y >= absN.z) {
                    surfaceCoords = vLocalPosition.xz;
                    effectiveOrigin = gridOriginMin.xz;
                    fragmentPlaneId = 1.0;
                    currentGridSize = max(gridSizeByPlane.y, 1e-6);
                    normalSide = geometricNormal.y >= 0.0 ? 1.0 : 0.0;
                  } else {
                    surfaceCoords = vLocalPosition.xy;
                    effectiveOrigin = gridOriginMin.xy;
                    fragmentPlaneId = 2.0;
                    currentGridSize = max(gridSizeByPlane.z, 1e-6);
                    normalSide = geometricNormal.z >= 0.0 ? 1.0 : 0.0;
                  }
                  const float cellEpsilon = 1e-4;

                  vec2 surfaceGridPos = (surfaceCoords - effectiveOrigin) / currentGridSize;
                  vec2 stableGridPos = surfaceGridPos + vec2(cellEpsilon);
                  vec2 cellCoords = floor(stableGridPos);
                  float fragmentCoord1 = cellCoords.x;
                  float fragmentCoord2 = cellCoords.y;
                  float fragmentCoord3 = normalSide;
                  
                  
                  // Verifica se esta célula está pintada (lê da textura)
                  bool isPainted = false;
                  if (useShaderPainting && numPaintedCells > 0) {
                    for (int i = 0; i < 700; i++) {
                      if (i >= numPaintedCells) break;
                      
                      // Lê coordenadas da textura
                      float u = (float(i) + 0.5) / 700.0;
                      vec4 texelData = texture2D(paintedCellsTexture, vec2(u, 0.5));
                      float paintedCoord1 = texelData.r;
                      float paintedCoord2 = texelData.g;
                      float paintedCoord3 = texelData.b;
                      float paintedPlaneId = texelData.a;
                      
                      // Só pinta se plano + coordenadas 2D forem iguais
                      bool match = abs(paintedPlaneId - fragmentPlaneId) < 0.1 &&
                                   abs(paintedCoord1 - fragmentCoord1) < 0.5 &&
                                   abs(paintedCoord2 - fragmentCoord2) < 0.5 &&
                                   abs(paintedCoord3 - fragmentCoord3) < 0.5;
                      
                      if (match) {
                        isPainted = true;
                        break;
                      }
                    }
                  }

                  // Linhas da grade no MESMO plano usado para identificar célula pintada.
                  // Isso elimina desalinhamento visual (fill x borda) em superfícies curvas.
                  vec2 lineGridPos = stableGridPos;
                  
                  vec2 gridFract = fract(lineGridPos);
                  vec2 gridEdgeDistance = min(gridFract, 1.0 - gridFract);
                  float lineWidth = 0.035;
                  float fwU = max(fwidth(lineGridPos.x), 1e-4);
                  float fwV = max(fwidth(lineGridPos.y), 1e-4);
                  float edgeU = 1.0 - smoothstep(lineWidth, lineWidth + fwU, gridEdgeDistance.x);
                  float edgeV = 1.0 - smoothstep(lineWidth, lineWidth + fwV, gridEdgeDistance.y);
                  float gridLine = max(edgeU, edgeV);
                  
                  // Aplica cor da célula pintada
                  if (isPainted) {
                    gl_FragColor.rgb = paintColor;
                  }
                  
                  // Overlay das linhas de grid
                  gl_FragColor.rgb = mix(gl_FragColor.rgb, gridColor, gridLine * gridOpacity);
                  `
                );
                
                // Guarda referência aos uniforms para atualização posterior
                mat.userData.shaderUniforms = shader.uniforms;
              };
              mat.userData.hasGridOverlay = true;
              mat.needsUpdate = true;
              console.log(`    ✅ Grid shader installed on material ${idx}`);
            } else {
              console.log(`    ⏭️ Grid shader already installed on material ${idx}`);
            }
          });
          
          // Marca materiais para recompilar no próximo frame sem alterar visibilidade.
          invalidate();
          console.log(`    🔄 Scheduled re-render after shader install`);
        }
      }
    });
    
    // Only proceed if meshes were found
    if (meshObjects.length === 0) return;
    
    // Mark as scanned to prevent re-running
    hasScannedMeshesRef.current = true;
    
    console.log(`🔍 Found ${meshes.length} meshes in visual group`);
    meshObjects.forEach((obj, idx) => {
      const name = obj.name || obj.parent?.name || "mesh";
      const matType = Array.isArray(obj.material) 
        ? `${obj.material.length} materials` 
        : obj.material?.type || "NONE";
      console.log(`  [${idx}] "${name}" (uuid: ${obj.uuid}, material: ${matType})`);
    });
    console.log(`✅ Mesh scan complete`);
    console.log(`🎨 Grid debug is ${SHOW_DEBUG_GRID ? "ENABLED" : "DISABLED"}`);
    console.log(`📐 Grid size: ${SURFACE_PIXEL_WORLD_SIZE}`);
    console.log(`🟢 Grid color: green (0x00ff00), opacity: 0.85`);

    if (!useColliderProxy) {
      onMeshRegistry?.(meshes);
    }
    onMeshesReady?.(meshObjects);
  });

  // Atualiza as células pintadas no shader
  useEffect(() => {
    if (!USE_SHADER_PAINTING || !paintedCells || !visualGroupRef.current) return;

    // Agrupa por partName (cada parte tem sua própria grid)
    const pixelsByPart = new Map<string, SurfacePixel[]>();
    paintedCells.forEach((pixel) => {
      const arr = pixelsByPart.get(pixel.partName) ?? [];
      arr.push(pixel);
      pixelsByPart.set(pixel.partName, arr);
    });
    
    visualGroupRef.current.traverse((obj) => {
      if (isMeshObject(obj) && obj.material) {
        const meshPartName = obj.uuid;
        const meshPixels = pixelsByPart.get(meshPartName) ?? [];

        const meshCells: number[] = [];
        const uniqueCells = new Set<string>();
        for (const pixel of meshPixels) {
          if (meshCells.length >= 700 * 4) break;

          const coord1 = pixel.planeCoord1;
          const coord2 = pixel.planeCoord2;
          const coord3 = pixel.planeCoord3;
          const planeId = pixel.plane === "yz" ? 0 : pixel.plane === "xz" ? 1 : 2;
          if (!Number.isFinite(planeId) || !Number.isFinite(coord1) || !Number.isFinite(coord2) || !Number.isFinite(coord3)) continue;

          const encoded = `${coord1}:${coord2}:${coord3}:${planeId}`;
          if (uniqueCells.has(encoded)) continue;
          uniqueCells.add(encoded);

          meshCells.push(coord1, coord2, coord3, planeId);
        }

        const cellsArray = meshCells.slice(0, 700 * 4);
        while (cellsArray.length < 700 * 4) {
          cellsArray.push(0, 0, 0, 0);
        }
        const numCells = Math.floor(meshCells.length / 4);

        const materials = Array.isArray(obj.material) ? obj.material : [obj.material];
        materials.forEach((mat: any) => {
          if (mat.userData.hasGridOverlay && mat.userData.shaderUniforms?.paintedCellsTexture) {
            const texture = mat.userData.shaderUniforms.paintedCellsTexture.value as THREE.DataTexture;
            if (texture.image?.data) {
              texture.image.data.set(new Float32Array(cellsArray));
              texture.needsUpdate = true;
              mat.userData.shaderUniforms.numPaintedCells.value = numCells;
            }
          }
        });
      }
    });
  }, [paintedCells]);

  return (
    <group position={[0, 0, 0]}>
      <group
        ref={visualGroupRef}
        onPointerMove={useColliderProxy ? undefined : onPointerMove}
        onPointerOut={useColliderProxy ? undefined : onPointerOut}
        onPointerDown={useColliderProxy ? undefined : onPointerDown}
      >
        {USE_COLLIDER_AS_VISUAL_TEST ? (
          <RenaultMasterCollider
            scale={0.02}
            rotation={[-Math.PI / 2, 0, 0]}
            position={[0, -2.5, 0]}
          />
        ) : (
          <RenaultMasterPanelVan
            scale={0.02}
            rotation={[-Math.PI / 2, 0, 0]}
            position={[0, -2.5, 0]}
          />
        )}
      </group>

      {useColliderProxy && (
        <ColliderProxyModel
          modelPath={colliderModelPath ?? COLLIDER_MODEL_PATH}
          onPointerMove={onPointerMove}
          onPointerOut={onPointerOut}
          onPointerDown={onPointerDown}
          onMeshRegistry={onMeshRegistry}
        />
      )}
    </group>
  );
}

function CameraBoundsGizmo() {
  const helper = useMemo(() => {
    const box = new THREE.Box3(
      new THREE.Vector3(CAMERA_BOUNDS.minX, CAMERA_BOUNDS.minY, CAMERA_BOUNDS.minZ),
      new THREE.Vector3(CAMERA_BOUNDS.maxX, CAMERA_BOUNDS.maxY, CAMERA_BOUNDS.maxZ),
    );

    const boxHelper = new THREE.Box3Helper(box, new THREE.Color("#38bdf8"));
    boxHelper.raycast = () => null;
    return boxHelper;
  }, []);

  useEffect(() => {
    return () => {
      helper.geometry.dispose();
      (helper.material as THREE.Material).dispose();
    };
  }, [helper]);

  return <primitive object={helper} />;
}


function FloorWithNormalMap() {
  const normalMap = useTexture("/asphalt.jpg");
  normalMap.wrapS = normalMap.wrapT = THREE.RepeatWrapping;
  normalMap.repeat.set(1, 1);

  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -2.51, 0]} receiveShadow>
      <planeGeometry args={[100, 100]} />
      <MeshReflectorMaterial
        blur={[400, 100]}
        resolution={1024}
        mixBlur={1}
        mixStrength={2}
        roughness={1}
        depthScale={1}
        minDepthThreshold={0.4}
        maxDepthThreshold={1.2}
        color="#0a0a0a"
        metalness={0.05}
        normalMap={normalMap}
        normalScale={new THREE.Vector2(0.1, 0.1)}
      />
    </mesh>
  );
}

function PixelGridMinimap({
  selectedMeshUuid,
  selectedMeshName,
  reservedCells,
}: {
  selectedMeshUuid: string | null;
  selectedMeshName: string;
  reservedCells: Set<string>;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const size = 220;
  const cellSize = Math.max(1, Math.floor(size / GRID_U));
  const width = GRID_U * cellSize;
  const height = GRID_V * cellSize;

  const occupiedBySelected = useMemo(() => {
    if (!selectedMeshUuid) return new Set<string>();

    const set = new Set<string>();
    for (const key of reservedCells) {
      const [partName, u, v] = key.split(":");
      if (partName === selectedMeshUuid && u !== undefined && v !== undefined) {
        set.add(`${u}:${v}`);
      }
    }
    return set;
  }, [reservedCells, selectedMeshUuid]);

  const occupiedCount = occupiedBySelected.size;
  const totalCells = GRID_U * GRID_V;
  const availableCount = totalCells - occupiedCount;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, width, height);

    // Base: available cells
    ctx.fillStyle = "#163025";
    ctx.fillRect(0, 0, width, height);

    // Occupied cells
    ctx.fillStyle = "#b91c1c";
    for (const id of occupiedBySelected) {
      const [uRaw, vRaw] = id.split(":");
      const u = Number(uRaw);
      const v = Number(vRaw);
      if (!Number.isFinite(u) || !Number.isFinite(v)) continue;
      ctx.fillRect(u * cellSize, v * cellSize, cellSize, cellSize);
    }

    // Major grid lines every 10 cells
    ctx.strokeStyle = "rgba(255,255,255,0.18)";
    ctx.lineWidth = 1;
    for (let i = 0; i <= GRID_U; i += 10) {
      const x = i * cellSize + 0.5;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let i = 0; i <= GRID_V; i += 10) {
      const y = i * cellSize + 0.5;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
  }, [occupiedBySelected, width, height, cellSize]);

  return (
    <div className="mt-3 rounded-md border border-white/20 bg-black/35 p-3">
      <p className="text-xs font-semibold uppercase tracking-wide text-white/90">Mapa UV</p>
      <p className="mt-1 text-xs text-white/70">Peca: {selectedMeshName}</p>
      <p className="mt-1 text-xs text-white/70">Disponiveis: {availableCount}</p>
      <p className="text-xs text-white/70">Ocupados: {occupiedCount}</p>
      <canvas ref={canvasRef} width={width} height={height} className="mt-2 w-full rounded border border-white/20" />
      <div className="mt-2 flex items-center gap-3 text-[11px] text-white/70">
        <span className="inline-flex items-center gap-1"><i className="h-2 w-2 rounded-sm bg-[#163025]" /> Livre</span>
        <span className="inline-flex items-center gap-1"><i className="h-2 w-2 rounded-sm bg-[#b91c1c]" /> Ocupado</span>
      </div>
    </div>
  );
}

function SurfacePixelOverlay({
  pixels,
  meshObjects,
}: {
  pixels: SurfacePixel[];
  meshObjects: Map<string, THREE.Mesh>;
}) {
  const meshByName = useMemo(() => {
    const map = new Map<string, THREE.Mesh>();
    for (const mesh of meshObjects.values()) {
      const name = getMeshDisplayName(mesh);
      if (!map.has(name)) map.set(name, mesh);
    }
    return map;
  }, [meshObjects]);

  const visiblePixels = useMemo(
    () => pixels.slice(-MAX_ACTIVE_SURFACE_DECALS),
    [pixels],
  );

  return (
    <group>
      {ENABLE_SURFACE_DECALS && visiblePixels.map((pixel) => {
        const targetMesh = meshObjects.get(pixel.meshUuid) ?? meshByName.get(pixel.meshName);
        if (!targetMesh) return null;
        return <SurfaceDecalMesh key={pixel.key} pixel={pixel} targetMesh={targetMesh} />;
      })}

      {SHOW_SURFACE_PIXEL_DEBUG_DOT && visiblePixels.map((pixel) => (
        <mesh key={pixel.key} position={pixel.position} renderOrder={9999}>
          <sphereGeometry args={[SURFACE_PIXEL_WORLD_SIZE * 0.18, 10, 10]} />
          <meshBasicMaterial color="#00e5ff" depthTest={false} depthWrite={false} />
        </mesh>
      ))}
    </group>
  );
}

export default function Page() {
  const controlsRef = useRef<CameraControlsRef | null>(null);
  const [hoveredCell, setHoveredCell] = useState<HoveredCell | null>(null);
  const [reservedCells, setReservedCells] = useState<Set<string>>(new Set());
  const [reservedSurfacePixels, setReservedSurfacePixels] = useState<Map<string, SurfacePixel>>(new Map());
  const [meshRegistry, setMeshRegistry] = useState<MeshInfo[]>([]);
  const [selectedMeshUuid, setSelectedMeshUuid] = useState<string | null>(null);
  const [forceSurfaceDecalFallback, setForceSurfaceDecalFallback] = useState(FORCE_SURFACE_DECAL_FALLBACK);
  const [lastSurfaceClick, setLastSurfaceClick] = useState<SurfaceClickDebug | null>(null);
  const lastHoverKeyRef = useRef<string | null>(null);
  const meshObjectsRef = useRef<Map<string, THREE.Mesh>>(new Map());
  const raycasterRef = useRef(new THREE.Raycaster());
  const cellCanvasByMeshRef = useRef<Map<string, HTMLCanvasElement>>(new Map());
  const cellTextureByMeshRef = useRef<Map<string, THREE.CanvasTexture>>(new Map());
  const reservedCellsRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    reservedCellsRef.current = reservedCells;
  }, [reservedCells]);

  const buildSurfacePixel = useCallback((e: ThreeEvent<PointerEvent>) => {
    if (!isMeshObject(e.object)) return null;

    const hitMesh = e.object;
    const hitMeshName = getMeshDisplayName(hitMesh);
    const faceIndexFromHit = e.faceIndex ?? 0;

    let targetMesh = meshObjectsRef.current.get(hitMesh.uuid);
    if (!targetMesh) {
      for (const candidate of meshObjectsRef.current.values()) {
        if (getMeshDisplayName(candidate) === hitMeshName) {
          targetMesh = candidate;
          break;
        }
      }
    }

    // Only do an extra raycast when picking happens on a proxy collider.
    const visualMeshes = Array.from(meshObjectsRef.current.values());
    let visualHit: THREE.Intersection | null = null;
    if (USE_SIMPLIFIED_COLLIDER_FOR_PICKING && visualMeshes.length > 0) {
      const raycaster = raycasterRef.current;
      raycaster.set(e.ray.origin, e.ray.direction);
      visualHit = raycaster.intersectObjects(visualMeshes, false)[0] ?? null;
    }

    const resolvedMesh = (visualHit?.object as THREE.Mesh | undefined) ?? targetMesh ?? hitMesh;
    const meshUuid = resolvedMesh.uuid;
    const meshName = getMeshDisplayName(resolvedMesh);
    const partName = meshUuid;
    const faceIndex = visualHit?.faceIndex ?? faceIndexFromHit;
    const projectedUv = visualHit?.uv ?? e.uv ?? null;

    const hitPoint = (visualHit?.point ?? e.point).clone();
    const localHitPoint = resolvedMesh.worldToLocal(hitPoint.clone());

    // Local face normal — fallback to mesh projection axis when raycast face data is missing.
    // This keeps selection working consistently after reloads/drivers that omit intersection face.
    let localFaceNormal = visualHit?.face?.normal?.clone() ?? e.face?.normal?.clone();
    if (!localFaceNormal) {
      const fallbackProjection = getMeshSurfaceProjection(resolvedMesh);
      localFaceNormal = new THREE.Vector3(
        fallbackProjection.normalAxis === "x" ? 1 : 0,
        fallbackProjection.normalAxis === "y" ? 1 : 0,
        fallbackProjection.normalAxis === "z" ? 1 : 0,
      );
    }
    localFaceNormal.normalize();

    if (!resolvedMesh.geometry.boundingBox) resolvedMesh.geometry.computeBoundingBox();
    const meshBoundingBox = resolvedMesh.geometry.boundingBox!;
    const hitProjection = projectionFromNormalAxis(getDominantNormalAxis(localFaceNormal), meshBoundingBox);
    const localCellSize = getCachedMeshLocalCellSize(resolvedMesh, hitProjection, SURFACE_PIXEL_WORLD_SIZE);
    const snapped = snapPointToSurfaceGrid(localHitPoint, hitProjection, localCellSize);
    const snappedWorldPoint = resolvedMesh.localToWorld(snapped.snappedPoint.clone());
    const normalSideCell = getAxisValue(localFaceNormal, hitProjection.normalAxis) >= 0 ? 1 : 0;

    // World normal oriented toward the camera (flip if face is back-facing relative to ray).
    const worldNormal = localFaceNormal.clone().transformDirection(resolvedMesh.matrixWorld).normalize();
    if (worldNormal.dot(e.ray.direction) > 0) worldNormal.negate();

    const decalPosition = snappedWorldPoint.clone().add(worldNormal.clone().multiplyScalar(SURFACE_PIXEL_OFFSET));
    const quat = surfaceQuaternionFromNormal(worldNormal);

    const plane = hitProjection.plane;
    const key = `${partName}:${plane}:${snapped.gridU}:${snapped.gridV}:${normalSideCell}`;

    return {
      pixel: {
        key,
        partName,
        meshUuid,
        meshName,
        plane,
        planeCoord1: snapped.gridU,
        planeCoord2: snapped.gridV,
        planeCoord3: normalSideCell,
        sharedPlane: undefined,
        sharedPlaneCoord1: undefined,
        sharedPlaneCoord2: undefined,
        faceIndex,
        gridU: snapped.gridU,
        gridV: snapped.gridV,
        uv: projectedUv ? [projectedUv.x, projectedUv.y] as [number, number] : undefined,
        position: [decalPosition.x, decalPosition.y, decalPosition.z] as [number, number, number],
        quaternion: [quat.x, quat.y, quat.z, quat.w] as [number, number, number, number],
      },
      key,
    };
  }, []);

  const ensureMeshCellTexture = useCallback((meshUuid: string) => {
    const existing = cellTextureByMeshRef.current.get(meshUuid);
    if (existing) return existing;

    const canvas = document.createElement("canvas");
    canvas.width = OVERLAY_WIDTH;
    canvas.height = OVERLAY_HEIGHT;
    const texture = new THREE.CanvasTexture(canvas);
    texture.flipY = false;
    texture.wrapS = THREE.ClampToEdgeWrapping;
    texture.wrapT = THREE.ClampToEdgeWrapping;
    texture.magFilter = THREE.NearestFilter;
    texture.minFilter = THREE.NearestFilter;
    texture.generateMipmaps = false;

    cellCanvasByMeshRef.current.set(meshUuid, canvas);
    cellTextureByMeshRef.current.set(meshUuid, texture);
    return texture;
  }, []);

  const paintMeshCellTexture = useCallback((meshUuid: string, source: Set<string>) => {
    const canvas = cellCanvasByMeshRef.current.get(meshUuid);
    const texture = cellTextureByMeshRef.current.get(meshUuid);
    if (!canvas || !texture) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Encontra a mesh para saber seu partName
    const mesh = meshObjectsRef.current.get(meshUuid);
    if (!mesh) return;
    const partName = mesh.uuid;

    ctx.clearRect(0, 0, OVERLAY_WIDTH, OVERLAY_HEIGHT);
    for (const key of source) {
      const [keyPartName, uRaw, vRaw] = key.split(":");
      if (keyPartName !== partName) continue;
      const u = Number(uRaw);
      const v = Number(vRaw);
      if (!Number.isFinite(u) || !Number.isFinite(v)) continue;

      ctx.fillStyle = "#ff2d55";
      // glTF materials use flipY=false, so keep V orientation aligned with the clicked UV.
      ctx.fillRect(
        u * OVERLAY_CELL_SIZE,
        v * OVERLAY_CELL_SIZE,
        OVERLAY_CELL_SIZE,
        OVERLAY_CELL_SIZE,
      );
    }

    texture.needsUpdate = true;
    invalidate();
  }, []);

  const paintMeshSurfaceTexture = useCallback((meshUuid: string, source: Map<string, SurfacePixel>) => {
    const canvas = cellCanvasByMeshRef.current.get(meshUuid);
    const texture = cellTextureByMeshRef.current.get(meshUuid);
    if (!canvas || !texture) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, OVERLAY_WIDTH, OVERLAY_HEIGHT);
    ctx.fillStyle = "#ff2d55";

    for (const pixel of source.values()) {
      if (pixel.meshUuid !== meshUuid || !pixel.uv) continue;

      const x = Math.round(pixel.uv[0] * (OVERLAY_WIDTH - OVERLAY_CELL_SIZE));
      const y = Math.round(pixel.uv[1] * (OVERLAY_HEIGHT - OVERLAY_CELL_SIZE));
      ctx.fillRect(x, y, OVERLAY_CELL_SIZE, OVERLAY_CELL_SIZE);
    }

    texture.needsUpdate = true;
    invalidate();
  }, []);

  const refreshAllMeshTextures = useCallback((source: Set<string>) => {
    for (const meshUuid of meshObjectsRef.current.keys()) {
      paintMeshCellTexture(meshUuid, source);
    }
  }, [paintMeshCellTexture]);

  const refreshAllSurfaceTextures = useCallback((source: Map<string, SurfacePixel>) => {
    for (const meshUuid of meshObjectsRef.current.keys()) {
      paintMeshSurfaceTexture(meshUuid, source);
    }
  }, [paintMeshSurfaceTexture]);

  const attachTextureToMeshMaterial = useCallback((mesh: THREE.Mesh, texture: THREE.CanvasTexture) => {
    cloneMeshMaterialsOnce(mesh);

    forEachStandardMaterial(mesh.material, (mat) => {
      // Strongly force overlay visibility for carpaint meshes
      if (mat.name === "carpaint" || mesh.name?.toLowerCase().includes("carpaint")) {
        mat.emissiveMap = texture;
        mat.emissive.set("#ffffff");
        mat.emissiveIntensity = 1.0;
        // Optionally assign to map for debug visibility
        // mat.map = texture;
      } else {
        mat.emissiveMap = texture;
        mat.emissive.set("#ff2d55");
        mat.emissiveIntensity = 0.8;
      }
      mat.needsUpdate = true;
    });
  }, []);

  const handleVanPointerMove = useCallback((e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation();
    if (PIXEL_MAPPING_MODE === "surface") {
      const surfaceHit = buildSurfacePixel(e);
      if (!surfaceHit) {
        setHoveredCell(null);
        lastHoverKeyRef.current = null;
        return;
      }

      if (lastHoverKeyRef.current === surfaceHit.key) return;
      lastHoverKeyRef.current = surfaceHit.key;

      setHoveredCell({
        meshUuid: surfaceHit.pixel.meshUuid,
        meshName: surfaceHit.pixel.meshName,
        faceIndex: surfaceHit.pixel.faceIndex,
        gridU: surfaceHit.pixel.gridU,
        gridV: surfaceHit.pixel.gridV,
        key: surfaceHit.key,
      });
      return;
    }

    if (!e.uv) {
      setHoveredCell(null);
      lastHoverKeyRef.current = null;
      return;
    }

    const mesh = e.object as THREE.Mesh;
    const meshUuid = mesh.uuid;
    const meshName = mesh.name || mesh.parent?.name || "mesh";
    const partName = meshUuid;
    const { u, v } = uvToCell(e.uv);
    const key = cellKey(partName, u, v);

    if (lastHoverKeyRef.current === key) return;
    lastHoverKeyRef.current = key;

    setHoveredCell({
      meshUuid,
      meshName,
      faceIndex: 0,
      gridU: u,
      gridV: v,
      key,
    });
  }, [buildSurfacePixel]);

  const handleVanPointerOut = useCallback(() => {
    setHoveredCell(null);
    lastHoverKeyRef.current = null;
  }, []);

  const handleVanPointerDown = useCallback((e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation();
    const mesh = e.object as THREE.Mesh;

    if (PIXEL_MAPPING_MODE === "surface") {
      const surfaceHit = buildSurfacePixel(e);
      if (!surfaceHit) return;
      setSelectedMeshUuid(surfaceHit.pixel.partName);

      setReservedSurfacePixels((prev) => {
        const next = new Map(prev);
        const wasSelected = next.has(surfaceHit.key);
        if (wasSelected) {
          next.delete(surfaceHit.key);
        } else {
          next.set(surfaceHit.key, surfaceHit.pixel);
          if (next.size > MAX_ACTIVE_SURFACE_DECALS) {
            const oldestKey = next.keys().next().value as string | undefined;
            if (oldestKey) next.delete(oldestKey);
          }
        }

        setLastSurfaceClick({
          meshName: surfaceHit.pixel.meshName,
          partName: surfaceHit.pixel.partName,
          plane: surfaceHit.pixel.plane,
          u: surfaceHit.pixel.planeCoord1,
          v: surfaceHit.pixel.planeCoord2,
          depth: surfaceHit.pixel.planeCoord3,
          key: surfaceHit.key,
          action: wasSelected ? "remove" : "add",
        });

        return next;
      });
      return;
    }

    if (!e.uv) return;

    const partName = mesh.uuid;
    setSelectedMeshUuid(partName);
    const { u, v } = uvToCell(e.uv);
    const key = cellKey(partName, u, v);

    setReservedCells((prev) => {
      const next = new Set(prev);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  }, [buildSurfacePixel]);

  const handleMeshRegistry = useCallback((meshes: MeshInfo[]) => {
    setMeshRegistry(meshes);
    setSelectedMeshUuid((prev) => prev ?? meshes[0]?.partName ?? null);
  }, []);

  const handleMeshesReady = useCallback((meshes: THREE.Mesh[]) => {
    const meshMap = new Map<string, THREE.Mesh>();
    for (const mesh of meshes) {
      meshMap.set(mesh.uuid, mesh);
      if (PIXEL_MAPPING_MODE === "uv") {
        const texture = ensureMeshCellTexture(mesh.uuid);
        attachTextureToMeshMaterial(mesh, texture);
      }
    }
    meshObjectsRef.current = meshMap;
    if (PIXEL_MAPPING_MODE === "uv") {
      refreshAllMeshTextures(reservedCellsRef.current);
    }
  }, [
    attachTextureToMeshMaterial,
    ensureMeshCellTexture,
    refreshAllMeshTextures,
  ]);

  useEffect(() => {
    if (PIXEL_MAPPING_MODE !== "uv") return;
    refreshAllMeshTextures(reservedCells);
  }, [reservedCells, refreshAllMeshTextures]);

  useEffect(() => {
    return () => {
      for (const texture of cellTextureByMeshRef.current.values()) {
        texture.dispose();
      }
      cellTextureByMeshRef.current.clear();
      cellCanvasByMeshRef.current.clear();
    };
  }, []);

  const selectedMeshName = useMemo(() => {
    if (!selectedMeshUuid) return "Nenhuma";
    return meshRegistry.find((m) => m.partName === selectedMeshUuid)?.name || selectedMeshUuid;
  }, [meshRegistry, selectedMeshUuid]);

  const totalRegisteredCells = meshRegistry.length * GRID_U * GRID_V;
  const surfacePixels = useMemo(() => Array.from(reservedSurfacePixels.values()), [reservedSurfacePixels]);
  const globalReservedCount = PIXEL_MAPPING_MODE === "uv" ? reservedCells.size : reservedSurfacePixels.size;
  const globalAvailableCount = PIXEL_MAPPING_MODE === "uv"
    ? Math.max(0, totalRegisteredCells - globalReservedCount)
    : null;

  const isHoveredReserved = hoveredCell
    ? (PIXEL_MAPPING_MODE === "uv" ? reservedCells.has(hoveredCell.key) : reservedSurfacePixels.has(hoveredCell.key))
    : false;
  const debugHoverText = ENABLE_UV_HOVER
    ? hoveredCell
      ? `Hover ${hoveredCell.meshName} -> face:${hoveredCell.faceIndex} grid:${hoveredCell.gridU},${hoveredCell.gridV} (${isHoveredReserved ? "ocupada" : "livre"})`
      : PIXEL_MAPPING_MODE === "uv"
        ? "Passe o mouse na van para ver a celula UV"
        : "Passe o mouse na van para ver a celula remapeada"
    : "Hover UV desativado para performance (clique segue ativo).";

  return (
    <main className="flex h-screen w-full flex-col bg-slate-950">
      <div className="pointer-events-none absolute left-10 top-10 z-10 text-white">
        <h1 className="text-4xl font-bold uppercase leading-none tracking-tighter">
          Van de Pixels
        </h1>
        <p className="mt-2 text-sm opacity-70">
          Use o botao esquerdo para girar e o direito para arrastar.
        </p>
        <p className="mt-2 text-xs opacity-80">
          {PIXEL_MAPPING_MODE === "uv"
            ? `Grid UV: ${GRID_U}x${GRID_V} | Reservadas: ${reservedCells.size}`
            : `Grid por face: ${PIXELS_PER_FACE_EDGE}x${PIXELS_PER_FACE_EDGE} | Reservadas: ${reservedSurfacePixels.size}`}
        </p>
        <p className="mt-1 text-xs opacity-80">{debugHoverText}</p>
      </div>

      <div className="absolute right-6 top-6 z-10 w-72 rounded-lg border border-white/20 bg-black/40 p-3 text-white backdrop-blur-sm">
        <p className="text-sm font-semibold uppercase tracking-wide">Painel de Pixels</p>
        <p className="mt-1 text-xs text-white/70">Registradas: {meshRegistry.length} pecas</p>
        <p className="text-xs text-white/70">
          Disponiveis (global): {globalAvailableCount === null ? "N/A (remap surface)" : globalAvailableCount}
        </p>
        <p className="text-xs text-white/70">Ocupados (global): {globalReservedCount}</p>

        <label className="mt-3 block text-xs text-white/80" htmlFor="mesh-selector">Peca para visualizar</label>
        <select
          id="mesh-selector"
          className="mt-1 w-full rounded border border-white/20 bg-black/50 px-2 py-1 text-xs"
          value={selectedMeshUuid ?? ""}
          onChange={(e) => setSelectedMeshUuid(e.target.value || null)}
        >
          {meshRegistry.map((mesh) => (
            <option key={mesh.uuid} value={mesh.partName}>
              {mesh.name}
            </option>
          ))}
        </select>

        {PIXEL_MAPPING_MODE === "uv" ? (
          <PixelGridMinimap
            selectedMeshUuid={selectedMeshUuid}
            selectedMeshName={selectedMeshName}
            reservedCells={reservedCells}
          />
        ) : (
          <div className="mt-3 rounded-md border border-white/20 bg-black/35 p-3">
            <p className="text-xs font-semibold uppercase tracking-wide text-white/90">Modo Remap Surface</p>
            <p className="mt-1 text-xs text-white/70">Pixels mapeados por face da mesh low-poly.</p>
            <p className="mt-1 text-xs text-white/70">Grade de {PIXELS_PER_FACE_EDGE}x{PIXELS_PER_FACE_EDGE} por triangulo.</p>
            {lastSurfaceClick && (
              <div className="mt-2 rounded border border-white/15 bg-black/30 p-2 text-[11px] text-white/75">
                <p>Ultimo clique: {lastSurfaceClick.action === "add" ? "ADD" : "REMOVE"}</p>
                <p>Mesh: {lastSurfaceClick.meshName}</p>
                <p>Plane: {lastSurfaceClick.plane} | Cell: {lastSurfaceClick.u}, {lastSurfaceClick.v}, side:{lastSurfaceClick.depth}</p>
              </div>
            )}
            <label className="mt-2 flex items-center gap-2 text-xs text-white/80">
              <input
                type="checkbox"
                checked={forceSurfaceDecalFallback}
                onChange={(e) => setForceSurfaceDecalFallback(e.target.checked)}
              />
              Forcar fallback visual (decal)
            </label>
          </div>
        )}
      </div>

      <Canvas
        shadows
        dpr={[1, 1.5]}
        performance={{ min: 0.5 }}
        frameloop="always"
        gl={{ antialias: true, toneMapping: THREE.ACESFilmicToneMapping }}
        camera={{ position: [10, 5, 15], fov: 35 }}
        onCreated={(state) => {
          state.gl.toneMappingExposure = 1.2;
          console.log("🎬 Canvas created, renderer ready");
        }}
      >

        {/* Fundo e Neblina alinhados ao Laranja do MMMV */}
        <color attach="background" args={["#f97316"]} /> {/* Laranja vibrante */}
        <fog attach="fog" args={["#f97316", 30, 60]} />



        {/* Luz Direcional simulando o Sol (Golden Hour) */}
        <directionalLight
          position={[5, 10, 5]}
          intensity={2.5}
          color="#fff4e0"
          castShadow
          shadow-mapSize={[2048, 2048]}
        />

        {/* Luz Ambiente mais baixa para dar contraste */}
        <ambientLight intensity={0.3} />

        {/* Luz de Preenchimento (Fill Light) para não deixar o outro lado preto */}
        <pointLight position={[-10, 5, 5]} intensity={1.5} color="#fcd34d" />

        <FloorWithNormalMap />

        <CameraControls
          ref={controlsRef}
          makeDefault
          dollyToCursor={false}
          verticalDragToForward={false}
          minDistance={5}
          maxDistance={15}
          azimuthRotateSpeed={0.5}
          polarRotateSpeed={0.5}
          dollySpeed={0.4}
          truckSpeed={0.8}
          mouseButtons={{
            left: 1,
            middle: 0,
            right: 2,
            wheel: 16,
          }}
        />

        <CameraRig controlsRef={controlsRef} />
        {SHOW_CAMERA_BOUNDS_GIZMO && <CameraBoundsGizmo />}

        <Suspense fallback={null}>
          <VanModel
            onPointerMove={ENABLE_UV_HOVER ? handleVanPointerMove : undefined}
            onPointerOut={ENABLE_UV_HOVER ? handleVanPointerOut : undefined}
            onPointerDown={handleVanPointerDown}
            onMeshRegistry={handleMeshRegistry}
            onMeshesReady={handleMeshesReady}
            useColliderProxy={false}
            colliderModelPath={COLLIDER_MODEL_PATH}
            paintedCells={reservedSurfacePixels}
          />
          {PIXEL_MAPPING_MODE === "surface" && (!USE_SHADER_PAINTING || forceSurfaceDecalFallback) && (
            <SurfacePixelOverlay
              pixels={surfacePixels}
              meshObjects={meshObjectsRef.current}
            />
          )}
          <Environment preset="sunset" environmentIntensity={0.8} />
        </Suspense>

        <ContactShadows
          position={[0, -2.48, 0]}
          opacity={0.5}
          scale={30}
          blur={1.5}
          far={4}
          frames={1}
        />
      </Canvas>
    </main>
  );
}