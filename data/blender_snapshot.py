'''
This script will take all OBJ files in a certain directory,
and one by one load them, render them, and remove them from the scene again.
https://blender.stackexchange.com/questions/80938/batch-rendering-of-10-000-images-from-10-000-obj-files
'''
import bpy
import pathlib, os

def adjust_camera():
    bpy.data.objects['Camera'].location[0] = 0 #locX
    bpy.data.objects['Camera'].location[1] = 0 #locY
    bpy.data.objects['Camera'].location[2] = 350 #locZ

    bpy.data.objects['Camera'].rotation_euler[0] = 0*3.14/180 #rotX
    bpy.data.objects['Camera'].rotation_euler[1] = 0*3.14/180 #rotY
    bpy.data.objects['Camera'].rotation_euler[2] = 0*3.14/180 #rotZ

    bpy.data.cameras['Camera'].lens = 25 #focalLength
    bpy.data.cameras['Camera'].clip_start = 50 #focalLength
    bpy.data.cameras['Camera'].clip_end = 500 #focalLength
def adjust_object(obj_fname, yaw_angle):
    bpy.data.objects[obj_fname].location[0] = 0 #locX
    bpy.data.objects[obj_fname].location[1] = 0 #locY
    bpy.data.objects[obj_fname].location[2] = 0 #locZ

    bpy.data.objects[obj_fname].rotation_euler[0] = -30*3.14/180 #rotX
    bpy.data.objects[obj_fname].rotation_euler[1] = yaw_angle*3.14/180 #rotY
    bpy.data.objects[obj_fname].rotation_euler[2] = 0*3.14/180 #rotZ


# Adjust this for where you have the OBJ files.
obj_root = pathlib.Path('/home/whitejet/Datasets/experiments/florenceface_subset')

# Before we start, make sure nothing is selected. The importer will select
# imported objects, which allows us to delete them after rendering.
bpy.ops.object.select_all(action='DESELECT')
render = bpy.context.scene.render

# desired yawing angles
yaw_angle = [-90,0,45,90]

for obj_fname in obj_root.glob('*.obj'):
    bpy.ops.import_scene.obj(filepath=str(obj_fname))

    obj_name=str(os.path.basename(obj_fname)).strip('.obj')
    for ang in yaw_angle:
        adjust_object(obj_name, yaw_angle=ang)
        adjust_camera()

        #set camera's view size
        render.resolution_x=900
        render.resolution_y=1000

        render.filepath = '//obj-%s_%d' % (obj_fname.stem,ang)
        bpy.ops.render.render(write_still=True)

        # Remember which meshes were just imported
        meshes_to_remove = []
        for ob in bpy.context.selected_objects:
            meshes_to_remove.append(ob.data)

    bpy.ops.object.delete()

    # Remove the meshes from memory too
    for mesh in meshes_to_remove:
        bpy.data.meshes.remove(mesh)
