"use client";

import { Suspense, useRef, useState, useEffect, useCallback, type ComponentRef, type MutableRefObject } from "react";
import * as THREE from "three";
import { Canvas, useFrame, type ThreeEvent } from "@react-three/fiber";
import {
  CameraControls,
  ContactShadows,
  Environment,
  MeshReflectorMaterial,
} from "@react-three/drei";
import { Model as RenaultVan } from "@/Renault_master_panel_van";

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
  maxDollySpeed: .5,
} as const;

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

export default function Page() {
  const controlsRef = useRef<CameraControlsRef | null>(null);
  const [hovered, setHovered] = useState<string | null>(null);
  const originalMaterialsRef = useRef<Map<string, THREE.Material>>(new Map());

  // Função para mudar o cursor (UX básica)
  useEffect(() => {
    document.body.style.cursor = hovered ? 'pointer' : 'auto';
  }, [hovered]);

  // Event handlers memoizados para performance
  const handlePointerOver = useCallback((e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation();
    const name = e.object.name || e.object.parent?.name || "Peça Oculta";
    setHovered(name);
  }, []);

  const handlePointerOut = useCallback(() => {
    setHovered(null);
  }, []);

  const handlePointerDown = useCallback((e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation();

    const target = e.object as THREE.Mesh;
    const partName = target.name || target.parent?.name || "Desconhecido";

    console.log("🎯 CLIQUE NA VAN DETECTADO!");
    console.log("Peça:", partName);
    console.log("ID do Objeto:", target.uuid);
    console.log("Ponto exato (World):", e.point);

    // Teste visual: muda a cor da peça clicada
    if (target.material) {
      // Salva o material original se ainda não foi salvo
      if (!originalMaterialsRef.current.has(target.uuid)) {
        originalMaterialsRef.current.set(target.uuid, target.material as THREE.Material);
      }

      // Clona e aplica a nova cor
      const material = target.material as THREE.MeshStandardMaterial;
      const newMaterial = material.clone();
      newMaterial.color.set('#ff0055');
      target.material = newMaterial;

      // Cleanup: descarta o material clonado após 2 segundos e restaura o original
      setTimeout(() => {
        const originalMaterial = originalMaterialsRef.current.get(target.uuid);
        if (originalMaterial) {
          target.material = originalMaterial;
          newMaterial.dispose();
        }
      }, 2000);
    }
  }, []);

  return (
    <main className="flex h-screen w-full flex-col bg-slate-950">
      <div className="pointer-events-none absolute left-10 top-10 z-10 text-white">
        <h1 className="text-4xl font-bold uppercase leading-none tracking-tighter">
          Van de Pixels
        </h1>
        <p className="mt-2 text-sm opacity-70">
          Use o botao esquerdo para girar e o direito para arrastar.
        </p>
      </div>

      <Canvas
        shadows
        dpr={[1, 1.5]}
        camera={{ position: [10, 5, 15], fov: 35 }}
        performance={{ min: 0.5 }}
        frameloop="demand"
      >
        <color attach="background" args={["#020617"]} />
        <fog attach="fog" args={["#020617", 22, 30]} />

        <ambientLight intensity={0.2} />
        <spotLight
          position={[10, 10, 10]}
          angle={0.5}
          penumbra={1}
          intensity={2}
          castShadow
        />

        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -2.51, 0]} receiveShadow>
          <planeGeometry args={[50, 50]} />
          <MeshReflectorMaterial
            blur={[100, 50]}
            resolution={256}
            mixBlur={0.8}
            mixStrength={20}
            roughness={1}
            depthScale={1}
            minDepthThreshold={0.5}
            maxDepthThreshold={1.2}
            color="#050505"
            metalness={0.3}
          />
        </mesh>

        <CameraControls
          ref={controlsRef}
          makeDefault
          dollyToCursor={false}
          verticalDragToForward={false}
          minDistance={3}
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

        <Suspense fallback={null}>
          <group
            position={[0, 0, 0]}
            onPointerOver={handlePointerOver}
            onPointerOut={handlePointerOut}
            onPointerDown={handlePointerDown}
          >
            <RenaultVan
              scale={0.02}
              rotation={[-Math.PI / 2, 0, 0]}
              position={[0, -2.5, 0]}
            />
          </group>
          <Environment preset="city" environmentIntensity={0.4} />
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