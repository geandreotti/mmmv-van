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
import { RenaultMasterCollider as RenaultVan } from "@/Renault_master_collider"; // Substitua pelo caminho correto do modelo

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
const SHOW_COLLIDER_VISUAL = true;
const COLLIDER_VISUAL_WIREFRAME = true;
const COLLIDER_VISUAL_OPACITY = 0.35;
const COLLIDER_VISUAL_COLOR = 0x00e5ff;
const COLLIDER_MODEL_PATH = "/renault_master_collider.glb";
const GRID_U = 50;
const GRID_V = 50;
const OVERLAY_CELL_SIZE = 16;
const OVERLAY_WIDTH = GRID_U * OVERLAY_CELL_SIZE;
const OVERLAY_HEIGHT = GRID_V * OVERLAY_CELL_SIZE;
const SURFACE_PIXEL_WORLD_SIZE = 0.25;
const SURFACE_PIXEL_OFFSET = 0.015;
const PIXELS_PER_FACE_EDGE = 12;
const ENABLE_SURFACE_DECALS = true;
const USE_SHADER_PAINTING = true; // Pintar direto no shader em vez de usar decals
const MAX_ACTIVE_SURFACE_DECALS = 700;
const SHOW_SURFACE_PIXEL_DEBUG_DOT = false;
const SHOW_DEBUG_GRID = true;
const MIN_SURFACE_ANGLE_DEGREES = 30; // Ângulo mínimo entre normal e plano do grid (evita células muito esticadas)
const MIN_PIXEL_DISTANCE = 0.15; // Distância mínima entre pixels em unidades mundiais (evita sobreposição)
const SHARED_CELL_NORMAL_DELTA = 0.2; // Se os dois eixos mais fortes da normal forem próximos, trata como célula compartilhada
const SHARED_CELL_NORMAL_RATIO = 0.72; // Também considera compartilhada quando o 2o eixo for forte em relação ao principal

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
};

type SurfacePixel = {
  key: string;
  meshUuid: string;
  meshName: string;
  plane: "xy" | "xz" | "yz";
  planeCoord1: number;
  planeCoord2: number;
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

type SurfaceProjection = {
  axisA: "x" | "y" | "z";
  axisB: "x" | "y" | "z";
  normalAxis: "x" | "y" | "z";
};

type SurfaceGridSnap = {
  snappedPoint: THREE.Vector3;
  gridU: number;
  gridV: number;
  gridN: number;
  worldGridX: number;
  worldGridY: number;
  worldGridZ: number;
  axisA: "x" | "y" | "z";
  axisB: "x" | "y" | "z";
  normalAxis: "x" | "y" | "z";
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

function clampCellIndex(value: number, max: number) {
  return Math.min(max - 1, Math.max(0, value));
}

function uvToCell(uv: THREE.Vector2) {
  const u = clampCellIndex(Math.floor(uv.x * GRID_U), GRID_U);
  const v = clampCellIndex(Math.floor(uv.y * GRID_V), GRID_V);
  return { u, v };
}

function cellKey(meshUuid: string, u: number, v: number) {
  return `${meshUuid}:${u}:${v}`;
}

function snapPointToSurfaceGrid(
  point: THREE.Vector3,
  normal: THREE.Vector3,
  cellSize: number,
): SurfaceGridSnap {
  const projection = surfaceProjectionFromNormal(normal);

  // Calcula índice da célula (qual célula contém o ponto)
  const worldGridX = Math.floor(point.x / cellSize);
  const worldGridY = Math.floor(point.y / cellSize);
  const worldGridZ = Math.floor(point.z / cellSize);

  // Coloca o ponto no CENTRO da célula
  const snappedPoint = new THREE.Vector3(
    (worldGridX + 0.5) * cellSize,
    (worldGridY + 0.5) * cellSize,
    (worldGridZ + 0.5) * cellSize
  );

  // Calcula grid local apenas para referência/debug
  const a = getAxisValue(snappedPoint, projection.axisA);
  const b = getAxisValue(snappedPoint, projection.axisB);
  const n = getAxisValue(snappedPoint, projection.normalAxis);

  const gridU = Math.floor(a / cellSize);
  const gridV = Math.floor(b / cellSize);
  const gridN = Math.floor(n / cellSize);

  return {
    snappedPoint,
    gridU,
    gridV,
    gridN,
    worldGridX,
    worldGridY,
    worldGridZ,
    axisA: projection.axisA,
    axisB: projection.axisB,
    normalAxis: projection.normalAxis,
  };
}

function surfaceProjectionFromNormal(normal: THREE.Vector3): SurfaceProjection {
  const absX = Math.abs(normal.x);
  const absY = Math.abs(normal.y);
  const absZ = Math.abs(normal.z);

  // Escolhe o eixo dominante de forma agressiva para estabilidade
  if (absX > absY && absX > absZ) {
    return { axisA: "y", axisB: "z", normalAxis: "x" };
  }
  if (absY > absX && absY > absZ) {
    return { axisA: "x", axisB: "z", normalAxis: "y" };
  }
  return { axisA: "x", axisB: "y", normalAxis: "z" };
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

function isPositionTooClose(
  position: [number, number, number],
  existingPixels: Map<string, SurfacePixel>,
  minDistance: number
): boolean {
  const pos = new THREE.Vector3(...position);
  for (const pixel of existingPixels.values()) {
    const existingPos = new THREE.Vector3(...pixel.position);
    if (pos.distanceTo(existingPos) < minDistance) {
      return true;
    }
  }
  return false;
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

function getMeshDisplayName(mesh: THREE.Mesh) {
  return mesh.name || mesh.parent?.name || "mesh";
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
      if (!(obj instanceof THREE.Mesh)) return;

      const toColliderMaterial = (mat: THREE.Material) => {
        if (SHOW_COLLIDER_VISUAL) {
          const visualMat = new THREE.MeshBasicMaterial({
            color: new THREE.Color(COLLIDER_VISUAL_COLOR),
            wireframe: COLLIDER_VISUAL_WIREFRAME,
            transparent: true,
            opacity: COLLIDER_VISUAL_OPACITY,
            depthWrite: false,
            depthTest: true,
            side: THREE.DoubleSide,
          });
          return visualMat;
        }

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
      if (obj instanceof THREE.Mesh) {
        const name = obj.name || obj.parent?.name || "mesh";
        meshes.push({ uuid: obj.uuid, name });
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

  useEffect(() => {
    console.log("[SHADER DEBUG] FIRST useEffect - Setting up shader on meshes");
    console.log("[SHADER DEBUG] visualGroupRef.current exists?", !!visualGroupRef.current);
    console.log("[SHADER DEBUG] SHOW_DEBUG_GRID?", SHOW_DEBUG_GRID);
    
    if (!visualGroupRef.current) {
      console.log("[SHADER DEBUG] !!! visualGroupRef.current is null, returning");
      return;
    }

    const meshObjects: THREE.Mesh[] = [];
    const meshes: MeshInfo[] = [];
    let meshFoundCount = 0;
    
    visualGroupRef.current.traverse((obj) => {
      if (obj.type === "Mesh") {
        meshFoundCount++;
        console.log(`[SHADER DEBUG] Found mesh #${meshFoundCount}:`, {
          name: obj.name || obj.uuid,
          hasMaterial: !!obj.material,
          materialType: obj.material?.constructor?.name,
          SHOW_DEBUG_GRID,
        });
        
        const name = obj.name || obj.parent?.name || "mesh";
        meshes.push({ uuid: obj.uuid, name });
        meshObjects.push(obj);
        
        // Aplica shader de grid debug se habilitado
        if (SHOW_DEBUG_GRID && obj.material) {
          console.log("[SHADER DEBUG] SHOW_DEBUG_GRID is true, obj.material exists");
          const materials = Array.isArray(obj.material) ? obj.material : [obj.material];
          
          materials.forEach((mat) => {
            if (!mat.userData.hasGridOverlay) {
              console.log("[SHADER DEBUG] Attaching grid overlay to mesh:", obj.name || obj.uuid);
              mat.onBeforeCompile = (shader) => {
                console.log("[SHADER DEBUG] !!! onBeforeCompile called - shader is being compiled");
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
                
                shader.uniforms.gridSize = { value: SURFACE_PIXEL_WORLD_SIZE };
                shader.uniforms.gridColor = { value: new THREE.Color(0x00ff00) };
                shader.uniforms.gridOpacity = { value: 0.3 };
                shader.uniforms.paintColor = { value: new THREE.Color(0xff2d55) };
                shader.uniforms.useShaderPainting = { value: USE_SHADER_PAINTING };
                shader.uniforms.paintedCellsTexture = { value: dataTexture };
                shader.uniforms.numPaintedCells = { value: 0 };
                
                shader.vertexShader = shader.vertexShader.replace(
                  '#include <common>',
                  `#include <common>
                  varying vec3 vWorldPosition;`
                );
                
                shader.vertexShader = shader.vertexShader.replace(
                  '#include <worldpos_vertex>',
                  `#include <worldpos_vertex>
                  vWorldPosition = (modelMatrix * vec4(position, 1.0)).xyz;`
                );
                
                shader.fragmentShader = shader.fragmentShader.replace(
                  '#include <common>',
                  `#include <common>
                  varying vec3 vWorldPosition;
                  uniform float gridSize;
                  uniform vec3 gridColor;
                  uniform float gridOpacity;
                  uniform vec3 paintColor;
                  uniform bool useShaderPainting;
                  uniform sampler2D paintedCellsTexture;
                  uniform int numPaintedCells;`
                );
                
                shader.fragmentShader = shader.fragmentShader.replace(
                  '#include <tonemapping_fragment>',
                  `#include <tonemapping_fragment>
                  
                  // Calcula posição na grid mundial
                  vec3 gridPos = vWorldPosition / gridSize;
                  // Índice da célula (qual célula contém este ponto)
                  vec3 cellCoords = floor(gridPos);
                                    // Determina o plano dominante da superfície
                                    vec3 absNormal = abs(normalize(cross(dFdx(vWorldPosition), dFdy(vWorldPosition))));
                                    // Prioridade de empate igual ao TypeScript: X > Y > Z
                                    bool isPlaneYZ = absNormal.x >= absNormal.y && absNormal.x >= absNormal.z;
                                    bool isPlaneXZ = !isPlaneYZ && absNormal.y >= absNormal.z;
                                    // isPlaneXY é o caso restante

                                    // Define o par de coordenadas 2D da célula para o plano atual
                                    float fragmentPlaneId = 2.0; // XY
                                    float fragmentCoord1 = cellCoords.x;
                                    float fragmentCoord2 = cellCoords.y;
                                    if (isPlaneYZ) {
                                      fragmentPlaneId = 0.0;
                                      fragmentCoord1 = cellCoords.y;
                                      fragmentCoord2 = cellCoords.z;
                                    } else if (isPlaneXZ) {
                                      fragmentPlaneId = 1.0;
                                      fragmentCoord1 = cellCoords.x;
                                      fragmentCoord2 = cellCoords.z;
                                    }
                  
                  
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
                      float paintedPlaneId = texelData.b;
                      
                      // Só pinta se plano + coordenadas 2D forem iguais
                      bool match = abs(paintedPlaneId - fragmentPlaneId) < 0.1 &&
                                   abs(paintedCoord1 - fragmentCoord1) < 0.01 &&
                                   abs(paintedCoord2 - fragmentCoord2) < 0.01;
                      
                      if (match) {
                        isPainted = true;
                        break;
                      }
                    }
                  }
                  
                  // Desenha linhas de grid apenas no plano dominante da superfície
                  vec3 gridFract = fract(gridPos);
                  float lineWidth = 0.02;
                  float gridLine = 0.0;
                  
                  // Desenha linhas apenas nos 2 eixos do plano dominante (ignora o eixo perpendicular)
                  if (isPlaneYZ) {
                    // Plano YZ (normal em X) - desenha apenas linhas Y e Z
                    if (gridFract.y < lineWidth || gridFract.y > 1.0 - lineWidth ||
                        gridFract.z < lineWidth || gridFract.z > 1.0 - lineWidth) {
                      gridLine = 1.0;
                    }
                  } else if (isPlaneXZ) {
                    // Plano XZ (normal em Y) - desenha apenas linhas X e Z
                    if (gridFract.x < lineWidth || gridFract.x > 1.0 - lineWidth ||
                        gridFract.z < lineWidth || gridFract.z > 1.0 - lineWidth) {
                      gridLine = 1.0;
                    }
                  } else {
                    // Plano XY (normal em Z) - desenha apenas linhas X e Y
                    if (gridFract.x < lineWidth || gridFract.x > 1.0 - lineWidth ||
                        gridFract.y < lineWidth || gridFract.y > 1.0 - lineWidth) {
                      gridLine = 1.0;
                    }
                  }
                  
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
              // Em frameloop='demand', força um frame para recompilar com onBeforeCompile.
              invalidate();
            }
          });
        }
      }
    });

    if (!useColliderProxy) {
      onMeshRegistry?.(meshes);
    }
    onMeshesReady?.(meshObjects);
    console.log("[SHADER DEBUG] First useEffect DONE - found", meshFoundCount, "meshes");
  }, [onMeshRegistry, onMeshesReady, useColliderProxy]);

  // ⚠️ MONITORAR CONTINUAMENTE: Attach shader quando mesh receber material
  useEffect(() => {
    if (!visualGroupRef.current) return;
    
    let intervalId: NodeJS.Timeout | null = null;
    let lastMeshCount = 0;
    
    const tryAttachShaders = () => {
      let currentMeshCount = 0;
      let attachedCount = 0;
      
      visualGroupRef.current?.traverse((obj) => {
        if (obj.type === "Mesh" && obj.material) {
          currentMeshCount++;
          
          const materials = Array.isArray(obj.material) ? obj.material : [obj.material];
          for (const mat of materials) {
            if (!mat.userData.hasGridOverlay && SHOW_DEBUG_GRID) {
              console.log("[SHADER DEBUG] ✓ ATTACHING SHADER NOW to mesh:", obj.name || obj.uuid);
              attachedCount++;
              
              mat.onBeforeCompile = (shader) => {
                console.log("[SHADER DEBUG] !!! onBeforeCompile called");
                const dataTexture = new THREE.DataTexture(
                  new Float32Array(700 * 4),
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
                
                shader.uniforms.gridSize = { value: SURFACE_PIXEL_WORLD_SIZE };
                shader.uniforms.gridColor = { value: new THREE.Color(0x00ff00) };
                shader.uniforms.gridOpacity = { value: 0.3 };
                shader.uniforms.paintColor = { value: new THREE.Color(0xff2d55) };
                shader.uniforms.useShaderPainting = { value: USE_SHADER_PAINTING };
                shader.uniforms.paintedCellsTexture = { value: dataTexture };
                shader.uniforms.numPaintedCells = { value: 0 };
                
                shader.vertexShader = shader.vertexShader.replace(
                  '#include <common>',
                  `#include <common>\n                  varying vec3 vWorldPosition;`
                );
                shader.vertexShader = shader.vertexShader.replace(
                  '#include <worldpos_vertex>',
                  `#include <worldpos_vertex>\n                  vWorldPosition = (modelMatrix * vec4(position, 1.0)).xyz;`
                );
                
                shader.fragmentShader = shader.fragmentShader.replace(
                  '#include <common>',
                  `#include <common>\n                  varying vec3 vWorldPosition;
                  uniform float gridSize;
                  uniform vec3 gridColor;
                  uniform float gridOpacity;
                  uniform vec3 paintColor;
                  uniform bool useShaderPainting;
                  uniform sampler2D paintedCellsTexture;
                  uniform int numPaintedCells;`
                );
                
                shader.fragmentShader = shader.fragmentShader.replace(
                  '#include <tonemapping_fragment>',
                  `#include <tonemapping_fragment>
                  vec3 gridPos = vWorldPosition / gridSize;
                  vec3 cellCoords = floor(gridPos);
                  vec3 absNormal = abs(normalize(cross(dFdx(vWorldPosition), dFdy(vWorldPosition))));
                  bool isPlaneYZ = absNormal.x >= absNormal.y && absNormal.x >= absNormal.z;
                  bool isPlaneXZ = !isPlaneYZ && absNormal.y >= absNormal.z;
                  float fragmentPlaneId = 2.0;
                  float fragmentCoord1 = cellCoords.x;
                  float fragmentCoord2 = cellCoords.y;
                  if (isPlaneYZ) {
                    fragmentPlaneId = 0.0;
                    fragmentCoord1 = cellCoords.y;
                    fragmentCoord2 = cellCoords.z;
                  } else if (isPlaneXZ) {
                    fragmentPlaneId = 1.0;
                    fragmentCoord1 = cellCoords.x;
                    fragmentCoord2 = cellCoords.z;
                  }
                  bool isPainted = false;
                  if (useShaderPainting && numPaintedCells > 0) {
                    for (int i = 0; i < 700; i++) {
                      if (i >= numPaintedCells) break;
                      float u = (float(i) + 0.5) / 700.0;
                      vec4 texelData = texture2D(paintedCellsTexture, vec2(u, 0.5));
                      bool match = abs(texelData.b - fragmentPlaneId) < 0.1 &&
                                   abs(texelData.r - fragmentCoord1) < 0.01 &&
                                   abs(texelData.g - fragmentCoord2) < 0.01;
                      if (match) {
                        isPainted = true;
                        break;
                      }
                    }
                  }
                  vec3 gridFract = fract(gridPos);
                  float lineWidth = 0.02;
                  float gridLine = 0.0;
                  if (isPlaneYZ) {
                    if (gridFract.y < lineWidth || gridFract.y > 1.0 - lineWidth ||
                        gridFract.z < lineWidth || gridFract.z > 1.0 - lineWidth) {
                      gridLine = 1.0;
                    }
                  } else if (isPlaneXZ) {
                    if (gridFract.x < lineWidth || gridFract.x > 1.0 - lineWidth ||
                        gridFract.z < lineWidth || gridFract.z > 1.0 - lineWidth) {
                      gridLine = 1.0;
                    }
                  } else {
                    if (gridFract.x < lineWidth || gridFract.x > 1.0 - lineWidth ||
                        gridFract.y < lineWidth || gridFract.y > 1.0 - lineWidth) {
                      gridLine = 1.0;
                    }
                  }
                  if (isPainted) {
                    gl_FragColor.rgb = paintColor;
                  }
                  gl_FragColor.rgb = mix(gl_FragColor.rgb, gridColor, gridLine * gridOpacity);`
                );
                
                mat.userData.shaderUniforms = shader.uniforms;
              };
              mat.userData.hasGridOverlay = true;
              mat.needsUpdate = true;
              invalidate();
            }
          }
        }
      });
      
      if (currentMeshCount > lastMeshCount) {
        console.log(`[SHADER DEBUG] Mesh count increased: ${lastMeshCount} → ${currentMeshCount}, attached: ${attachedCount}`);
        lastMeshCount = currentMeshCount;
      }
    };
    
    tryAttachShaders();
    intervalId = setInterval(tryAttachShaders, 100);
    
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [invalidate]);

  // ⚠️ Registrar meshes no dropdown quando tiverem shader
  useEffect(() => {
    if (!visualGroupRef.current || !onMeshRegistry) return;
    
    let lastRegisteredCount = 0;
    let intervalId: NodeJS.Timeout | null = null;
    
    const registerMeshesWithShaders = () => {
      const meshesToRegister: MeshInfo[] = [];
      
      visualGroupRef.current?.traverse((obj) => {
        if (obj.type === "Mesh" && obj.material) {
          const materials = Array.isArray(obj.material) ? obj.material : [obj.material];
          for (const mat of materials) {
            if (mat.userData.hasGridOverlay) {
              const name = obj.name || obj.parent?.name || "mesh";
              meshesToRegister.push({ uuid: obj.uuid, name });
              break;
            }
          }
        }
      });
      
      if (meshesToRegister.length > lastRegisteredCount) {
        console.log("[SHADER DEBUG] ✓✓✓ Registering", meshesToRegister.length, "meshes in dropdown");
        lastRegisteredCount = meshesToRegister.length;
        onMeshRegistry(meshesToRegister);
      }
    };
    
    registerMeshesWithShaders();
    intervalId = setInterval(registerMeshesWithShaders, 200);
    
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [onMeshRegistry]);

  // Atualiza as células pintadas no shader
  useEffect(() => {
    console.log("[SHADER DEBUG] paintedCells update:", {
      USE_SHADER_PAINTING,
      paintedCellsSize: paintedCells?.size,
      hasVisualGroupRef: !!visualGroupRef.current,
      paintedCellsNull: paintedCells == null,
    });
    
    if (!USE_SHADER_PAINTING || !paintedCells || !visualGroupRef.current) {
      console.log("[SHADER DEBUG] Early return - one of conditions failed");
      return;
    }

    // Agrupa por mesh para evitar vazamento de pintura entre malhas diferentes.
    const encodedCellsByMesh = new Map<string, Set<string>>();
    console.log("[SHADER DEBUG] Starting to process", paintedCells.size, "painted cells");

    const addEncodedCell = (
      meshUuid: string,
      plane: "xy" | "xz" | "yz",
      coord1: number,
      coord2: number,
    ) => {
      if (!Number.isFinite(coord1) || !Number.isFinite(coord2)) return;

      const planeId = plane === "yz" ? 0 : plane === "xz" ? 1 : 2;
      const encoded = `${planeId}:${coord1}:${coord2}`;
      const current = encodedCellsByMesh.get(meshUuid) ?? new Set<string>();
      current.add(encoded);
      encodedCellsByMesh.set(meshUuid, current);
    };

    paintedCells.forEach((pixel) => {
      const fallbackCellX = Math.floor(pixel.position[0] / SURFACE_PIXEL_WORLD_SIZE);
      const fallbackCellY = Math.floor(pixel.position[1] / SURFACE_PIXEL_WORLD_SIZE);
      const fallbackCellZ = Math.floor(pixel.position[2] / SURFACE_PIXEL_WORLD_SIZE);

      const plane = pixel.plane ?? "xy";
      const planeCoord1 = Number.isFinite(pixel.planeCoord1)
        ? pixel.planeCoord1
        : (plane === "yz" ? fallbackCellY : fallbackCellX);
      const planeCoord2 = Number.isFinite(pixel.planeCoord2)
        ? pixel.planeCoord2
        : (plane === "xy" ? fallbackCellY : fallbackCellZ);

      addEncodedCell(pixel.meshUuid, plane, planeCoord1, planeCoord2);

      if (pixel.sharedPlane) {
        const sharedPlane = pixel.sharedPlane;
        const sharedCoord1 = Number.isFinite(pixel.sharedPlaneCoord1)
          ? pixel.sharedPlaneCoord1
          : (sharedPlane === "yz" ? fallbackCellY : fallbackCellX);
        const sharedCoord2 = Number.isFinite(pixel.sharedPlaneCoord2)
          ? pixel.sharedPlaneCoord2
          : (sharedPlane === "xy" ? fallbackCellY : fallbackCellZ);
        addEncodedCell(pixel.meshUuid, sharedPlane, sharedCoord1, sharedCoord2);
      }
    });
    
    const syncPaintDataToShader = () => {
      console.log("[SHADER DEBUG] SyncPaintDataToShader called");
      console.log("[SHADER DEBUG] visualGroupRef.current exists?", !!visualGroupRef.current);
      
      let didUpdateShaderData = false;
      let hasPendingShaderCompile = false;
      let meshCount = 0;

      visualGroupRef.current?.traverse((obj) => {
        if (obj.type === "Mesh") {
          console.log("[SHADER DEBUG] Found Mesh:", {
            name: obj.name,
            typeCheck: obj.type === "Mesh",
            hasMaterial: !!obj.material,
            materialType: obj.material?.constructor?.name,
            isMaterial: obj.material instanceof THREE.Material,
          });
        }
        
        if (obj.type === "Mesh" && obj.material) {
          meshCount++;
          
          // ⚠️ FIRST TIME SETUP: Se o shader ainda não foi attachado, faz agora!
          const materials = Array.isArray(obj.material) ? obj.material : [obj.material];
          materials.forEach((mat) => {
            if (!mat.userData.hasGridOverlay && SHOW_DEBUG_GRID) {
              console.log("[SHADER DEBUG] !!! LATE SHADER ATTACHMENT - mesh just loaded, attaching now");
              mat.onBeforeCompile = (shader) => {
                console.log("[SHADER DEBUG] !!! onBeforeCompile called - shader is being compiled");
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
                
                shader.uniforms.gridSize = { value: SURFACE_PIXEL_WORLD_SIZE };
                shader.uniforms.gridColor = { value: new THREE.Color(0x00ff00) };
                shader.uniforms.gridOpacity = { value: 0.3 };
                shader.uniforms.paintColor = { value: new THREE.Color(0xff2d55) };
                shader.uniforms.useShaderPainting = { value: USE_SHADER_PAINTING };
                shader.uniforms.paintedCellsTexture = { value: dataTexture };
                shader.uniforms.numPaintedCells = { value: 0 };
                
                shader.vertexShader = shader.vertexShader.replace(
                  '#include <common>',
                  `#include <common>
                  varying vec3 vWorldPosition;`
                );
                
                shader.vertexShader = shader.vertexShader.replace(
                  '#include <worldpos_vertex>',
                  `#include <worldpos_vertex>
                  vWorldPosition = (modelMatrix * vec4(position, 1.0)).xyz;`
                );
                
                shader.fragmentShader = shader.fragmentShader.replace(
                  '#include <common>',
                  `#include <common>
                  varying vec3 vWorldPosition;
                  uniform float gridSize;
                  uniform vec3 gridColor;
                  uniform float gridOpacity;
                  uniform vec3 paintColor;
                  uniform bool useShaderPainting;
                  uniform sampler2D paintedCellsTexture;
                  uniform int numPaintedCells;`
                );
                
                shader.fragmentShader = shader.fragmentShader.replace(
                  '#include <tonemapping_fragment>',
                  `#include <tonemapping_fragment>
                  
                  // Calcula posição na grid mundial
                  vec3 gridPos = vWorldPosition / gridSize;
                  // Índice da célula (qual célula contém este ponto)
                  vec3 cellCoords = floor(gridPos);
                                    // Determina o plano dominante da superfície
                                    vec3 absNormal = abs(normalize(cross(dFdx(vWorldPosition), dFdy(vWorldPosition))));
                                    // Prioridade de empate igual ao TypeScript: X > Y > Z
                                    bool isPlaneYZ = absNormal.x >= absNormal.y && absNormal.x >= absNormal.z;
                                    bool isPlaneXZ = !isPlaneYZ && absNormal.y >= absNormal.z;
                                    // isPlaneXY é o caso restante

                                    // Define o par de coordenadas 2D da célula para o plano atual
                                    float fragmentPlaneId = 2.0; // XY
                                    float fragmentCoord1 = cellCoords.x;
                                    float fragmentCoord2 = cellCoords.y;
                                    if (isPlaneYZ) {
                                      fragmentPlaneId = 0.0;
                                      fragmentCoord1 = cellCoords.y;
                                      fragmentCoord2 = cellCoords.z;
                                    } else if (isPlaneXZ) {
                                      fragmentPlaneId = 1.0;
                                      fragmentCoord1 = cellCoords.x;
                                      fragmentCoord2 = cellCoords.z;
                                    }
                  
                  
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
                      float paintedPlaneId = texelData.b;
                      
                      // Só pinta se plano + coordenadas 2D forem iguais
                      bool match = abs(paintedPlaneId - fragmentPlaneId) < 0.1 &&
                                   abs(paintedCoord1 - fragmentCoord1) < 0.01 &&
                                   abs(paintedCoord2 - fragmentCoord2) < 0.01;
                      
                      if (match) {
                        isPainted = true;
                        break;
                      }
                    }
                  }
                  
                  // Desenha linhas de grid apenas no plano dominante da superfície
                  vec3 gridFract = fract(gridPos);
                  float lineWidth = 0.02;
                  float gridLine = 0.0;
                  
                  // Desenha linhas apenas nos 2 eixos do plano dominante (ignora o eixo perpendicular)
                  if (isPlaneYZ) {
                    // Plano YZ (normal em X) - desenha apenas linhas Y e Z
                    if (gridFract.y < lineWidth || gridFract.y > 1.0 - lineWidth ||
                        gridFract.z < lineWidth || gridFract.z > 1.0 - lineWidth) {
                      gridLine = 1.0;
                    }
                  } else if (isPlaneXZ) {
                    // Plano XZ (normal em Y) - desenha apenas linhas X e Z
                    if (gridFract.x < lineWidth || gridFract.x > 1.0 - lineWidth ||
                        gridFract.z < lineWidth || gridFract.z > 1.0 - lineWidth) {
                      gridLine = 1.0;
                    }
                  } else {
                    // Plano XY (normal em Z) - desenha apenas linhas X e Y
                    if (gridFract.x < lineWidth || gridFract.x > 1.0 - lineWidth ||
                        gridFract.y < lineWidth || gridFract.y > 1.0 - lineWidth) {
                      gridLine = 1.0;
                    }
                  }
                  
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
              console.log("[SHADER DEBUG] Late shader attachment - set mat.needsUpdate = true");
            }
          });
          
          console.log(`[SHADER DEBUG] Mesh #${meshCount}:`, {
            uuid: obj.uuid,
            hasGridOverlay: materials[0]?.userData?.hasGridOverlay,
            hasShaderUniforms: !!materials[0]?.userData?.shaderUniforms,
            paintedCellsTexture: !!materials[0]?.userData?.shaderUniforms?.paintedCellsTexture,
          });
          
          const encodedCells = encodedCellsByMesh.get(obj.uuid) ?? new Set<string>();
          console.log(`[SHADER DEBUG] Mesh #${meshCount} encoded cells:`, encodedCells.size);
          
          const meshCells: number[] = [];
          for (const encoded of encodedCells) {
            if (meshCells.length >= 700 * 4) break;
            const [planeIdRaw, coord1Raw, coord2Raw] = encoded.split(":");
            const planeId = Number(planeIdRaw);
            const coord1 = Number(coord1Raw);
            const coord2 = Number(coord2Raw);
            if (!Number.isFinite(planeId) || !Number.isFinite(coord1) || !Number.isFinite(coord2)) continue;
            meshCells.push(coord1, coord2, planeId, 1.0);
          }

          const cellsArray = meshCells.slice(0, 700 * 4);
          while (cellsArray.length < 700 * 4) {
            cellsArray.push(0, 0, 0, 0);
          }
          const numCells = Math.floor(meshCells.length / 4);

          materials.forEach((mat: any) => {
            if (!mat.userData.hasGridOverlay) {
              console.log("[SHADER DEBUG] Material has no grid overlay");
              return;
            }

            if (!mat.userData.shaderUniforms?.paintedCellsTexture) {
              console.log("[SHADER DEBUG] !!! SHADER NOT COMPILED YET - marking for retry");
              hasPendingShaderCompile = true;
              mat.needsUpdate = true;
              return;
            }

            console.log("[SHADER DEBUG] Updating texture with", numCells, "cells");
            const texture = mat.userData.shaderUniforms.paintedCellsTexture.value as THREE.DataTexture;
            texture.image.data.set(new Float32Array(cellsArray));
            texture.needsUpdate = true;
            mat.userData.shaderUniforms.numPaintedCells.value = numCells;
            didUpdateShaderData = true;
          });
        }
      });

      if (didUpdateShaderData || hasPendingShaderCompile) {
        console.log("[SHADER DEBUG] Calling invalidate():", {
          didUpdateShaderData,
          hasPendingShaderCompile,
          meshCountProcessed: meshCount,
        });
        invalidate();
      } else {
        console.log("[SHADER DEBUG] !!! No shader data updated and no pending compile - meshCount:", meshCount);
      }

      return hasPendingShaderCompile;
    };

    const needsRetry = syncPaintDataToShader();
    console.log("[SHADER DEBUG] Sync result - needsRetry:", needsRetry);
    
    let rafId: number | null = null;
    if (needsRetry) {
      console.log("[SHADER DEBUG] Scheduling RAF retry...");
      rafId = window.requestAnimationFrame(() => {
        console.log("[SHADER DEBUG] RAF callback - retrying sync");
        syncPaintDataToShader();
      });
    }

    return () => {
      if (rafId !== null) {
        window.cancelAnimationFrame(rafId);
      }
    };
  }, [paintedCells]);

  return (
    <group position={[0, 0, 0]}>
      <group
        ref={visualGroupRef}
        onPointerMove={useColliderProxy ? undefined : onPointerMove}
        onPointerOut={useColliderProxy ? undefined : onPointerOut}
        onPointerDown={useColliderProxy ? undefined : onPointerDown}
      >
        <RenaultVan
          scale={0.02}
          rotation={[-Math.PI / 2, 0, 0]}
          position={[0, -2.5, 0]}
        />
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
      const [meshUuid, u, v] = key.split(":");
      if (meshUuid === selectedMeshUuid && u !== undefined && v !== undefined) {
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
    if (!e.face || e.faceIndex == null) return null;

    const hitMesh = e.object as THREE.Mesh;
    const hitMeshName = getMeshDisplayName(hitMesh);
    const faceIndexFromHit = e.faceIndex;

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
    const faceIndex = visualHit?.faceIndex ?? faceIndexFromHit;
    const projectedUv = visualHit?.uv ?? e.uv ?? null;

    const projectedPoint = (visualHit?.point ?? e.point).clone();

    const baseNormal = visualHit?.face?.normal ?? e.face.normal;
    const worldNormal = baseNormal.clone().transformDirection(resolvedMesh.matrixWorld).normalize();
    
    // Snap no plano dominante da superfície para manter células quadradas e alinhadas.
    const snapped = snapPointToSurfaceGrid(projectedPoint, worldNormal, SURFACE_PIXEL_WORLD_SIZE);

    // Recalcula normal na posição snapped fazendo raycast da posição do grid
    const raycaster = raycasterRef.current;
    const rayOrigin = snapped.snappedPoint.clone().add(worldNormal.clone().multiplyScalar(0.5));
    const rayDirection = worldNormal.clone().negate();
    raycaster.set(rayOrigin, rayDirection);
    
    const snappedHit = raycaster.intersectObject(resolvedMesh, false)[0];
    const finalNormal = snappedHit?.face 
      ? snappedHit.face.normal.clone().transformDirection(resolvedMesh.matrixWorld).normalize()
      : worldNormal;

    // Valida o ângulo da superfície para evitar células muito esticadas/sobrepostas
    if (!isSurfaceAngleValid(finalNormal)) {
      return null; // Rejeita posicionamento em superfícies muito inclinadas
    }

    // Apply a tiny offset to avoid z-fighting while keeping the decal attached to surface.
    const decalPosition = snapped.snappedPoint.clone().add(finalNormal.clone().multiplyScalar(SURFACE_PIXEL_OFFSET));

    const quat = surfaceQuaternionFromNormal(finalNormal);
    
    // Cria chave usando apenas os 2 eixos do plano dominante (ignora o eixo perpendicular à superfície)
    // Isso faz as células "colapsarem" ao longo do eixo perpendicular, unindo-as
    const absX = Math.abs(finalNormal.x);
    const absY = Math.abs(finalNormal.y);
    const absZ = Math.abs(finalNormal.z);

    const planeCandidates: Array<{
      axis: "x" | "y" | "z";
      value: number;
      plane: "xy" | "xz" | "yz";
      coord1: number;
      coord2: number;
    }> = [
      {
        axis: "x",
        value: absX,
        plane: "yz",
        coord1: snapped.worldGridY,
        coord2: snapped.worldGridZ,
      },
      {
        axis: "y",
        value: absY,
        plane: "xz",
        coord1: snapped.worldGridX,
        coord2: snapped.worldGridZ,
      },
      {
        axis: "z",
        value: absZ,
        plane: "xy",
        coord1: snapped.worldGridX,
        coord2: snapped.worldGridY,
      },
    ].sort((a, b) => b.value - a.value);
    
    let plane: "xy" | "xz" | "yz";
    let planeCoord1: number;
    let planeCoord2: number;
    plane = planeCandidates[0].plane;
    planeCoord1 = planeCandidates[0].coord1;
    planeCoord2 = planeCandidates[0].coord2;

    let sharedPlane: "xy" | "xz" | "yz" | undefined;
    let sharedPlaneCoord1: number | undefined;
    let sharedPlaneCoord2: number | undefined;

    const secondaryCandidate = planeCandidates[1];
    const isSharedCell = secondaryCandidate
      ? ((planeCandidates[0].value - secondaryCandidate.value) <= SHARED_CELL_NORMAL_DELTA ||
          (secondaryCandidate.value / Math.max(planeCandidates[0].value, 1e-6)) >= SHARED_CELL_NORMAL_RATIO)
      : false;

    if (isSharedCell && secondaryCandidate) {
      sharedPlane = secondaryCandidate.plane;
      sharedPlaneCoord1 = secondaryCandidate.coord1;
      sharedPlaneCoord2 = secondaryCandidate.coord2;
    }

    const key = `${meshUuid}:${plane}:${planeCoord1}:${planeCoord2}`;
    
    return {
      pixel: {
        key,
        meshUuid,
        meshName,
        plane,
        planeCoord1,
        planeCoord2,
        sharedPlane,
        sharedPlaneCoord1,
        sharedPlaneCoord2,
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

    ctx.clearRect(0, 0, OVERLAY_WIDTH, OVERLAY_HEIGHT);
    for (const key of source) {
      const [keyMeshUuid, uRaw, vRaw] = key.split(":");
      if (keyMeshUuid !== meshUuid) continue;
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
    if (!mesh.userData.pixelMaterialCloned) {
      if (Array.isArray(mesh.material)) {
        mesh.material = mesh.material.map((mat) => mat.clone());
      } else {
        mesh.material = mesh.material.clone();
      }
      mesh.userData.pixelMaterialCloned = true;
    }

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
    const { u, v } = uvToCell(e.uv);
    const key = cellKey(meshUuid, u, v);

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
      setSelectedMeshUuid(surfaceHit.pixel.meshUuid);

      setReservedSurfacePixels((prev) => {
        const next = new Map(prev);
        if (next.has(surfaceHit.key)) {
          next.delete(surfaceHit.key);
        } else {
          // Verifica se não está muito próximo de outro pixel (evita sobreposição)
          if (isPositionTooClose(surfaceHit.pixel.position, prev, MIN_PIXEL_DISTANCE)) {
            return prev; // Não adiciona
          }
          
          next.set(surfaceHit.key, surfaceHit.pixel);
          if (next.size > MAX_ACTIVE_SURFACE_DECALS) {
            const oldestKey = next.keys().next().value as string | undefined;
            if (oldestKey) next.delete(oldestKey);
          }
        }
        return next;
      });
      return;
    }

    if (!e.uv) return;

    setSelectedMeshUuid(mesh.uuid);

    const { u, v } = uvToCell(e.uv);
    const key = cellKey(mesh.uuid, u, v);

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
    setSelectedMeshUuid((prev) => prev ?? meshes[0]?.uuid ?? null);
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
    return meshRegistry.find((m) => m.uuid === selectedMeshUuid)?.name || "Peca";
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
            <option key={mesh.uuid} value={mesh.uuid}>
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
          </div>
        )}
      </div>

      <Canvas
        shadows
        dpr={[1, 1.5]}
        performance={{ min: 0.5 }}
        frameloop="demand"
        gl={{ antialias: true, toneMapping: THREE.ACESFilmicToneMapping }}
        camera={{ position: [10, 5, 15], fov: 35 }}
        onCreated={(state) => {
          state.gl.toneMappingExposure = 1.2;
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
            useColliderProxy={USE_SIMPLIFIED_COLLIDER_FOR_PICKING}
            colliderModelPath={COLLIDER_MODEL_PATH}
            paintedCells={reservedSurfacePixels}
          />
          {PIXEL_MAPPING_MODE === "surface" && !USE_SHADER_PAINTING && (
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