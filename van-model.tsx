import { useGLTF, OrbitControls, Stage } from '@react-three/drei'
import { Canvas } from '@react-three/fiber'

function Model() {
  const { scene } = useGLTF('/minha-van.glb')
  return <primitive object={scene} />
}

export default function VanScene() {
  return (
    <div className="h-screen w-full bg-slate-900">
      <Canvas dpr={[1, 2]} camera={{ position: [0, 0, 5], fov: 45 }}>
        <color attach="background" args={['#101010']} />
        <Stage environment="city" intensity={0.6}>
           <Model />
        </Stage>
        <OrbitControls makeDefault />
      </Canvas>
    </div>
  )
}