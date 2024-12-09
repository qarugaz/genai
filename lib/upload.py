import io
from datetime import datetime

from google.cloud import storage

def upload_image(
        image,
        bucket_name,
        prompt=None,
        additional_metadata=None
):
    """
    Upload a Stable Diffusion generated image directly to GCP Cloud Storage

    Args:
        image (PIL.Image): Generated image from Stable Diffusion
        bucket_name (str): GCP Cloud Storage bucket name
        prompt (str, optional): Prompt used to generate the image
        additional_metadata (dict, optional): Extra metadata to attach

    Returns:
        str: Public URL of the uploaded image (if bucket is public)
    """
    # Initialize GCP Storage client
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Generate a unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Sanitize prompt for filename
    if prompt:
        safe_prompt = "".join(
            char for char in prompt
            if char.isalnum() or char in [' ', '_']
        )[:50].replace(' ', '_')
        blob_name = f"stable_diffusion/{timestamp}_{safe_prompt}.png"
    else:
        blob_name = f"stable_diffusion/{timestamp}_image.png"

    # Create a blob and upload the image
    blob = bucket.blob(blob_name)

    # Save image to a bytes buffer
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    # Upload the image
    blob.upload_from_file(
        img_byte_arr,
        content_type='image/png'
    )

    # Optional: Add metadata
    metadata = {
        'generated_timestamp': timestamp
    }
    if prompt:
        metadata['prompt'] = prompt

    # Add any additional metadata
    if additional_metadata:
        metadata.update(additional_metadata)

    # Set metadata
    blob.metadata = metadata
    blob.patch()

    # Optional: Make public (remove if you want private)
    # blob.make_public()

    print(f"Image uploaded to {blob_name}")

    # Return public URL if the blob is public
    return blob.public_url if blob.public_url else None


# Batch upload example
def upload_multiple_images(images, bucket_name, prompt=None):
    """
    Upload multiple images from a single generation batch
    """
    urls = []
    for idx, image in enumerate(images):
        metadata = {'batch_index': idx} if len(images) > 1 else None
        url = upload_image(
            image,
            bucket_name,
            prompt=prompt,
            additional_metadata=metadata
        )
        urls.append(url)

    return urls
